import { Success, Response, ServerError } from '../model/response.model';
import { PythonProcess } from '../helpers/python-process';
import { withTimeMeasure, Path, TimeMeasureResult } from '../helpers/util';
import { ImageProcessor } from '../helpers/image-processor';
import { KerasObjectDetector, ObjectDetector } from '../model/object-detection/object-detector';
import { Detection } from '../model/object-detection/object-detection.model';
import { ImageDetection } from '../model/common-interface.model';

export class Controller {

  async classifyImage(data: { image: string }): Promise<Response<ImageDetection>> {
    const image = data.image;
    const pythonScript = Path.pathFromRelativePath('app/model/python-scripts/runnable.py');
    try {
      const objectDetector: ObjectDetector = new KerasObjectDetector(pythonScript);
      const result: TimeMeasureResult<Detection[]> = await withTimeMeasure(() => objectDetector.processImageBase64(image));
      const detectionResult: ImageDetection = { objects: result.result, executionTime: result.time, image: {base64: image} };
      return Promise.resolve(new Success(detectionResult));
    }
    catch ( err ) {
      return Promise.reject(new ServerError(err));
    }
  }

  async classifyRandomImage(): Promise<Response<ImageDetection | null>> {
    const image = await ImageProcessor.readImageInBase64('assets/testimage2.png');
    const pythonScript = Path.pathFromRelativePath('app/model/python-scripts/runnable.py');
    try {
      const objectDetector: ObjectDetector = new KerasObjectDetector(pythonScript);
      const result: TimeMeasureResult<Detection[]> = await withTimeMeasure(() => objectDetector.processImageBase64(image));
      const detectionResult: ImageDetection = { objects: result.result, executionTime: result.time, image: {base64: image} };
      return Promise.resolve(new Success(detectionResult));
    }
    catch ( err ) {
      return Promise.reject(new ServerError(err));
    }
  }

  private sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
