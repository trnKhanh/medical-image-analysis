export const formatFileSize = (bytes: number): string => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
};

export const formatDate = (timestamp: number): string => {
    return new Date(timestamp * 1000).toLocaleString();
};

export const downloadFile = (blob: Blob, filename: string): void => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
};

export const createImageFromBackground = (background: number[][][]): string => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.createImageData(background[0].length, background.length);

    for (let y = 0; y < background.length; y++) {
        for (let x = 0; x < background[0].length; x++) {
            const pixelIndex = (y * background[0].length + x) * 4;
            imageData.data[pixelIndex] = background[y][x][0];
            imageData.data[pixelIndex + 1] = background[y][x][1];
            imageData.data[pixelIndex + 2] = background[y][x][2];
            imageData.data[pixelIndex + 3] = background[y][x][3];
        }
    }

    canvas.width = background[0].length;
    canvas.height = background.length;
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL().split(',')[1];
};
