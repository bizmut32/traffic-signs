import { Success, Response } from '../model/exceptions.model';
import { ClassificationResult } from '../model/common-interface';
import { readImageInBase64 } from '../model/image-reader.model';
const path = require('path');

export class Controller {
  hello (name: String): Promise<Response> {
    return new Promise (async (resolve) => {
      setTimeout(() => {
        resolve (new Success(`Hello ${name}!`));
      }, 1000);
    });
  }

  classifyImage(data: { image: string }): Promise<Response<ClassificationResult>> {
    return new Promise ((resolve, reject) => {
      const guesses: number[] = [];
      for (let i = 0; i < 43; ++i)
        guesses.push(Math.random() * 0.8);
      guesses[1] = 0.98345;

      resolve(new Success({image: {base64: data.image}, guesses, executionTime: 123}));
    });
  }

  classifyRandomImage(): Promise<Response<ClassificationResult>> {
    return new Promise (async (resolve, reject) => {
      const guesses: number[] = [];
      for (let i = 0; i < 43; ++i)
        guesses.push(Math.random() * 0.8);
      guesses[1] = 0.98345;

      const image = await readImageInBase64('assets/40.jpg');
      await this.sleep(1000);

      resolve(new Success({image: {base64: image}, guesses, executionTime: 123}));
    });
  }

  private sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
