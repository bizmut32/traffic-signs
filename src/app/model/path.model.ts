export class Path {
    static pathFromRelativePath(relativePath: string): string {
        const appRoot = require('app-root-path');
        return `${appRoot}/${relativePath}`;
    }

    static fileUrlFromRelativePath(relativePath: string): string {
        return `file://${this.pathFromRelativePath(relativePath)}`;
    }
}
