import { Path } from './util';
import * as fs from 'fs';

export class ImageProcessor {
    static readImageInBase64(fileName: string): Promise<string> {
        const base64Img = require('base64-img');

        return new Promise((resolve, reject) => {
            const filePath = Path.pathFromRelativePath(fileName);
            base64Img.base64(filePath, (err: any, data: string) => {
                if (err) return reject(err);
                resolve(data);
            });
        });
    }

    static base64ToPng(base64: string, dest: string): Promise<string> {
        const destPath = Path.pathFromRelativePath(dest);
        const base64Data = base64.replace(/^data:image\/png;base64,/, '');

        return new Promise((resolve, reject) => {
            fs.writeFile(destPath, base64Data, 'base64', (err: any) => {
                if (err) return reject(err);
                else resolve(destPath);
            });
        });
    }

    static deleteFile(path: string): Promise<void> {
        return new Promise (resolve => {
            fs.unlink(path, () => Promise.resolve());
        });
    }
}
