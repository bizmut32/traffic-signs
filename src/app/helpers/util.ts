export const argmax = (arr: number[]): number => arr.reduce((acc, element, i) => arr[acc] > element ? i : acc);
export const argmaxObj = (obj: { [key: number]: number }): number => {
    let max = 0;
    for (const key in obj) if (!(max in obj) || obj[key] > obj[max]) max = parseInt(key);
    return max;
};
