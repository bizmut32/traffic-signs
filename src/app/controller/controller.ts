import { Success, Response, ServerError } from '../model/response.model';
import { ClassificationResult } from '../model/common-interface.model';
import { PythonProcess } from '../helpers/python-process';
import { withTimeMeasure, Path } from '../helpers/util';
import { ImageProcessor } from '../helpers/image-processor';
import { KerasObjectDetector, ObjectDetector } from '../model/object-detection/object-detector';
import { Detection } from '../model/object-detection/object-detection.model';

export class Controller {

  async classifyImage(data: { image: string }): Promise<Response<Detection[]>> {
    const image = data.image;
    const pythonScript = Path.pathFromRelativePath('app/model/python-scripts/runnable.py');
    const objectDetector: ObjectDetector = new KerasObjectDetector(pythonScript);
    const result = await withTimeMeasure(() => objectDetector.processImageBase64(image));
    return Promise.resolve(new Success(result));
  }

  classifyRandomImage(): Promise<Response<ClassificationResult>> {
    return new Promise (async (resolve, reject) => {
      const guesses: number[] = [];
      for (let i = 0; i < 43; ++i)
        guesses.push(Math.random() * 0.8);
      guesses[1] = 0.98345;

      const image = await ImageProcessor.readImageInBase64('assets/40.jpg');
      await this.sleep(1000);

      resolve(new Success({image: {base64: image}, guesses, executionTime: 123}));
    });
  }


  private sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
