import { Detection, numbersToBoundingBox } from './object-detection.model';
import { ImageProcessor } from '../../helpers/image-processor';
import { Process, PythonProcess, ZeroRPCProcess } from '../../helpers/python-process';
import { ClassificationOutput, GRTBSClassification } from './classification.model';

type Detections = Promise<Detection[]>;

export class MultipleDetectionsError extends Error {
    constructor (message: string | null = null) {
       super(message ?? 'Cannot detect multiple images with the same object');
    }
}

export class OutputMismatchException extends Error {
    constructor (message: string | null = null) {
       super(message ?? 'Output type did not match to expected');
    }
}

export interface ObjectDetector {
    processImageBase64(base64: string): Detections;
    processImage(path: string): Detections;
}

export class KerasObjectDetector implements ObjectDetector {
    private readonly imageFolder: string = `assets/tmp`;
    private imagePath: string | null = null;

    constructor (private pythonScript: string) {}

    processImageBase64(base64: string): Detections {
        return new Promise(resolve => {
            this.generateFilePath();
            ImageProcessor.base64ToPng(base64, this.imagePath ?? '').then(path => {
                const result = this.processImage(path);
                result.finally(() => { ImageProcessor.deleteFile(path); });
                resolve(result);
            });
        });
    }

    processImage(path: string): Detections {
        const process: Process = new ZeroRPCProcess(this.pythonScript);
        const detection = process.start(path);
        return detection
            .then(result => this.resultToBoundingBoxes(result));
    }

    private resultToBoundingBoxes(str: string): Detection[] {
        const output: ClassificationOutput = JSON.parse(str);
        const detections: Detection[] = [];
        for (let i = 0; i < output.bounding_boxes.length; ++i) {
            const bbox = output.bounding_boxes[i];
            const x1 = bbox[0], x2 = bbox[1], y1 = bbox[2], y2 = bbox[3];
            detections.push({
                boundingBox: numbersToBoundingBox(x1, x2, y1, y2),
                classification: new GRTBSClassification(output.classifications[i]),
                certainty: output.certainties[i]
            });
        }
        return detections;
    }

    generateFilePath() {
        if (this.imagePath !== null)
            throw new MultipleDetectionsError();
        this.imagePath = `${this.imageFolder}/image_${Math.round(Math.random() * 100000)}.png`;
    }
}

