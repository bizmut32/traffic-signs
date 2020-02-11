import { Success, Response } from '../model/exceptions.model';

export class Controller {
  hello (name: String): Promise<Response> {
    return new Promise (async (resolve) => {
      setTimeout(() => {
        resolve (new Success(`Hello ${name}!`));
      }, 1000);
    });
  }
}
