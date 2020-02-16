import * as fs from 'fs';
import * as path from 'path';
const base64Img = require('base64-img');

export function readImageInBase64(fileName: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const filePath = `${__dirname}/../../${fileName}`;
    base64Img.base64(filePath, (err: any, data: string) => {
      if (err) return reject(err);
      resolve(data);
    });
  });
}
