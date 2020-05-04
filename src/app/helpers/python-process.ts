import { Path } from './util';
import * as fs from 'fs';

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
            // setTimeout(() => { return reject('Python process timeout (10s)'); }, 10_000);
            pythonProcess.stdout.on('data', (data: any) => {
                const res = data.toString().slice(0, -1);
                if (res === 'error') reject('Something went wrong');
                else resolve(res);
            });

            pythonProcess.stderr.on('error', (error: any) => { reject(error); });
        });
    }
}
