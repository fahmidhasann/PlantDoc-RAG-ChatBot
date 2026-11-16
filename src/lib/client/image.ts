/**
 * Prepares an uploaded photo for the API.
 *
 * A phone camera file is routinely 6-12 MB, which is larger than a serverless
 * request body is allowed to be once base64 adds a third on top. Downscaling in
 * the browser keeps the upload small and costs nothing in diagnostic value:
 * leaf lesions are legible well below 1600px, and the model charges by image
 * tokens, not by megabytes.
 */

const MAX_EDGE = 1600;
const TARGET_BYTES = 2.4 * 1024 * 1024; // leaves headroom under the 3 MB cap

export interface PreparedImage {
  /** base64 without the data: prefix */
  data: string;
  mimeType: string;
  /** object URL for the local preview; revoke when the message is discarded */
  previewUrl: string;
  width: number;
  height: number;
}

export async function prepareImage(file: File): Promise<PreparedImage> {
  if (!file.type.startsWith('image/')) {
    throw new Error('That file is not an image. Use a JPEG, PNG or WebP photo.');
  }

  const bitmap = await createImageBitmap(file).catch(() => {
    throw new Error('That image could not be read. Try a JPEG or PNG.');
  });

  const scale = Math.min(1, MAX_EDGE / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('This browser could not process the image.');
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();

  // Step the quality down until it fits; JPEG keeps photographic detail best.
  let blob: Blob | null = null;
  for (const quality of [0.86, 0.74, 0.62, 0.5]) {
    blob = await new Promise<Blob | null>(resolve =>
      canvas.toBlob(resolve, 'image/jpeg', quality)
    );
    if (blob && blob.size <= TARGET_BYTES) break;
  }
  if (!blob) throw new Error('That image could not be processed.');
  if (blob.size > TARGET_BYTES) {
    throw new Error('That image is too large even after compression. Try a smaller photo.');
  }

  const buffer = await blob.arrayBuffer();
  return {
    data: toBase64(buffer),
    mimeType: 'image/jpeg',
    previewUrl: URL.createObjectURL(blob),
    width,
    height,
  };
}

function toBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  // Chunked so a large image does not blow the argument limit of fromCharCode.
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}
