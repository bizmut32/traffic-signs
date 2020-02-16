
export interface ClassifiedImage {
  base64: string;
}

export interface ClassificationResult {
  image: ClassifiedImage;
  guesses: number[];
  executionTime: number;
}

export const signNames: string[] = [
  '30 km/h',
  '40 km/h'
];
