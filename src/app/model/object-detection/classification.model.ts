import { labels } from './labels';

export interface ClassificationOutput {
    bounding_boxes: number[][];
    classifications: number[];
    certainties: number[];
}

export interface Classification {
    readonly serial: number;
    readonly label: string;
}

export class GRTBSClassification implements Classification {
    constructor(public serial: number) {}
    label: string = labels[this.serial];
}
