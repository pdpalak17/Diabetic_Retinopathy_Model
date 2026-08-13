import { customAlphabet } from "nanoid";

const caseSuffix = customAlphabet("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", 5);
const allowedImageTypes = new Set(["image/jpeg", "image/png"]);

export type ParsedScreeningImage = {
  bytes: Buffer;
  mimeType: "image/jpeg" | "image/png";
  extension: "jpg" | "png";
};

export function buildCaseCode(date = new Date()): string {
  const datePart = date.toISOString().slice(0, 10).replaceAll("-", "");
  return `SCR-${datePart}-${caseSuffix()}`;
}

export function parseScreeningImage(dataUrl: string): ParsedScreeningImage | null {
  const match = /^data:(image\/(?:jpeg|png));base64,([A-Za-z0-9+/=\s]+)$/.exec(dataUrl);
  if (!match || !allowedImageTypes.has(match[1])) return null;

  const bytes = Buffer.from(match[2].replace(/\s/g, ""), "base64");
  if (bytes.length === 0 || bytes.length > 6 * 1024 * 1024) return null;

  const mimeType = match[1] as ParsedScreeningImage["mimeType"];
  return { bytes, mimeType, extension: mimeType === "image/png" ? "png" : "jpg" };
}
