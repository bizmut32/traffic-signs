import { TensorflowModelBase, Model } from './cnn-base';
import { Image } from './image';
import { argmax } from '../../helpers/util';

export class Cnn implements Model {
    private model: TensorflowModelBase;

    constructor(name: string, private preprocessor: (img: Image) => Image) {
        this.model = new TensorflowModelBase(name);
    }

    predict(input: Image): number {
        const preprocessed = this.preprocessor(input);
        const results = this.model.predict(preprocessed);
        const result = argmax(results);
        return result;
    }
}
