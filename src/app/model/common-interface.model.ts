import { Detection } from './object-detection/object-detection.model';

export interface ImageDetection {
  objects: Detection[];
  image: { base64: string };
  executionTime: number;
}
