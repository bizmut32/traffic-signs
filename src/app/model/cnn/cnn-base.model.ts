import { LayersModel, loadLayersModel, Tensor, tensor2d } from '@tensorflow/tfjs';
import { Image } from '../image';
import { Path } from '../path.model';

export interface Model {
    predict(input: Image): any;
}

export class OutputShapeMismatchException extends Error {
    constructor(obj: any) {
        super('Output shape does not match expected');
        console.log(`>>>>>>`, obj);
    }
}

export class TensorflowModelBase implements Model {
    protected model: LayersModel;

    constructor(name: string) {
        const file = this.filePath(name);
        loadLayersModel(file).then(model => this.model = model);
    }

    predict(input: Image): number[] {
        const inputTensor = tensor2d(input.pixels);
        const result: Tensor | Tensor[] = this.model.predict(inputTensor);
        if (result instanceof Tensor)
            return Array.from(result.dataSync());
        throw new OutputShapeMismatchException(result);
    }

    private filePath(name: string): string {
        return `${Path.fileUrlFromRelativePath('assets/models')}/${name}/model.json`;
    }
}
