import { describe, expect, it } from "vitest";
import { reviewOutputSchema, screeningInputSchema } from "./routers/screening";
import { buildCaseCode, parseScreeningImage } from "./screening";

describe("screening intake helpers", () => {
  it("creates a legible date-prefixed screening identifier", () => {
    const code = buildCaseCode(new Date("2026-08-13T00:00:00.000Z"));
    expect(code).toMatch(/^SCR-20260813-[A-Z0-9]{5}$/);
  });

  it("accepts a supported retinal image payload and preserves its type", () => {
    const parsed = parseScreeningImage("data:image/png;base64,aGVsbG8=");
    expect(parsed?.mimeType).toBe("image/png");
    expect(parsed?.extension).toBe("png");
    expect(parsed?.bytes.toString()).toBe("hello");
  });

  it("rejects unsupported image formats", () => {
    expect(parseScreeningImage("data:image/gif;base64,aGVsbG8=")).toBeNull();
  });

  it("requires the clinical intake fields used by the workflow", () => {
    const result = screeningInputSchema.safeParse({
      patientId: "PT-204",
      patientAge: 58,
      imageName: "fundus.png",
      imageData: `data:image/png;base64,${"a".repeat(100)}`,
    });
    expect(result.success).toBe(true);
  });

  it("accepts a complete, source-attributed review output", () => {
    const result = reviewOutputSchema.safeParse({
      id: 4,
      severity: "moderate",
      confidence: 92,
      visualFindings: "Validated model output identified a reviewable retinal pattern.",
      riskLevel: "elevated",
      riskScore: 68,
      groundedExplanation: "This recorded summary combines the supplied model outputs with a source-attributed evidence passage for qualified professional review.",
      evidence: [{
        sourceTitle: "Clinical guideline excerpt",
        publisher: "Clinical evidence source",
        sourceType: "guideline",
        excerpt: "A source-attributed passage that supports the recorded explanation and can be inspected by the clinician.",
        relevanceScore: 91,
      }],
    });
    expect(result.success).toBe(true);
  });
});
