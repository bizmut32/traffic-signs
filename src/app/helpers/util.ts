export const argmax = (arr: number[]): number => arr.reduce((acc, element, i) => arr[acc] > element ? i : acc);
export const argmaxObj = (obj: { [key: number]: number }): number => {
    let max = 0;
    for (const key in obj) if (!(max in obj) || obj[key] > obj[max]) max = parseInt(key);
    return max;
};

export function withTimeMeasure<T>(callback: () => Promise<T>): Promise<T> {
    const start = Date.now();
    const res = callback();
    res.finally(() => {
        const end = Date.now();
        console.log('Process took', end - start, 'ms');
    });
    return res;
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




