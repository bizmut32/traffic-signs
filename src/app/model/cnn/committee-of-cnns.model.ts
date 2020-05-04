import { Model } from './cnn-base.model';
import { Cnn } from './cnn.model';
import { Image } from '../image';
import { argmaxObj } from '../../helpers/util';

class CommitteeOfCnns implements Model {
    private committee: Cnn[];

    constructor() {
        const models: string[] = [0, 1, 2, 3, 4].map(i => `committee-${i}`);
        this.committee = models.map(name => new Cnn(name, (e) => e));
    }

    predict(input: Image): number {
        const votes = this.committee.map<number>(cnn => cnn.predict(input));
        const voting: { [key: number]: number } = {};
        votes.forEach(vote => {
            if (!(vote in voting)) voting[vote] = 1;
            else voting[vote]++;
        });
        return argmaxObj(voting);
    }
}

export type ImageClassifier = CommitteeOfCnns;
