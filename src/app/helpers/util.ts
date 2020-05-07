export const argmax = (arr: number[]): number => arr.reduce((acc, element, i) => arr[acc] > element ? i : acc);
export const argmaxObj = (obj: { [key: number]: number }): number => {
    let max = 0;
    for (const key in obj) if (!(max in obj) || obj[key] > obj[max]) max = parseInt(key);
    return max;
};

export interface TimeMeasureResult<T> {
    result: T;
    time: number;
}

export function withTimeMeasure<T>(callback: () => Promise<T>): Promise<TimeMeasureResult<T>> {
    const start = Date.now();
    const res = callback();
    return res.then(result => {
        const end = Date.now();
        return { result, time: end - start };
    });
}

export class Path {
    static pathFromRelativePath(relativePath: string): string {
        const appRoot = require('app-root-path');
        return `${appRoot}/src/${relativePath}`;
    }

    static fileUrlFromRelativePath(relativePath: string): string {
        return `file://${this.pathFromRelativePath(relativePath)}`;
    }
}




