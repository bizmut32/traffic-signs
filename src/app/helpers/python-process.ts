import { Path } from './util';
import * as fs from 'fs';
const zerorpc = require('zerorpc');

export interface Process {
    start(...args: any): Promise<string>;
}

export class PythonProcess implements Process {
    file: string;

    constructor(fileName: string) {
        this.file = fileName;
    }

    start(...args: any): Promise<string> {
        if (!fs.existsSync(this.file))
            return Promise.reject(this.file + ' not found');

        const spawn = require('child_process').spawn;
        const pythonProcess = spawn('python', [this.file, ...args]);

        return new Promise<string>((resolve, reject) => {
            let result = false;
            setTimeout(() => {
                if (result) return;
                result = true;
                return reject('Python process timeout (10s)');
            }, 12_000);

            pythonProcess.stdout.on('data', (data: any) => {
                result = true;
                const res = data.toString().slice(0, -1);
                if (res === 'error') reject('Something went wrong');
                else resolve(res);
            });

            pythonProcess.stderr.on('error', (error: any) => {
                result = true;
                reject(error);
            });
        });
    }
}

export class ZeroRPCProcess implements Process {
    file: string;

    constructor(fileName: string) {
        this.file = fileName;
    }

    start(...args: any): Promise<string> {
        if (!fs.existsSync(this.file))
            return Promise.reject(this.file + ' not found');
        const client = new zerorpc.Client();
        client.connect('tcp://127.0.0.1:4242');
        return new Promise((resolve, reject) => {
            let result = false;
            setTimeout(() => {
                if (result) return;
                result = true;
                return reject('Python process timeout (10s)');
            }, 12_000);

            client.invoke('detect', ...args, function(error: any, res: any, more: any) {
                console.log(res);
                if (error) reject(error);
                else resolve(res);
            });
        });
    }
}