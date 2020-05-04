import { Path } from './path.model';

export class PythonProcess {
    file: string;

    constructor(fileName: string) {
        this.file = this.filePath(fileName);
    }

    start(...args: any): Promise<string> {
        const spawn = require('child_process').spawn;
        const pythonProcess = spawn('python', [this.file, ...args]);

        return new Promise<string>((resolve, reject) => {
            pythonProcess.stdout.on('data', (data: any) => {
                const res = data.toString().slice(0, -1);
                if (res === 'error') reject('Something went wrong');
                else resolve(res);
            });
            pythonProcess.stderr.on('error', (error: any) => { reject(error); });
        });
    }

    private filePath(name: string): string {
        const scriptsFolder = 'src/scripts';
        return `${Path.pathFromRelativePath(scriptsFolder)}/${name}`;
    }
}
