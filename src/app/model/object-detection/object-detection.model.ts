import { Classification } from './classification.model';

export interface Coordinates {
    x: number;
    y: number;
}

type Corner = Coordinates;

export function numbersToBoundingBox(x1: number, x2: number, y1: number, y2: number) {
    return {
        topLeft: { x: x1, y: y1 },
        bottomRight: { x: x2, y: y2 }
    };
}

export interface BoundingBox {
    topLeft: Corner;
    bottomRight: Corner;
}

export interface Detection {
    boundingBox: BoundingBox;
    classification: Classification;
}
