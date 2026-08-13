import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { createScreeningCase, getScreeningCaseForUser, listScreeningCasesForUser, saveScreeningReview } from "../db";
import { buildCaseCode, parseScreeningImage } from "../screening";
import { storagePut } from "../storage";
import { protectedProcedure, router } from "../_core/trpc";

export const screeningInputSchema = z.object({
  patientId: z.string().trim().min(2, "Enter a patient identifier.").max(80),
  patientAge: z.number().int().min(18).max(120),
  diabetesDurationYears: z.number().min(0).max(100).optional(),
  hba1c: z.number().min(0).max(25).optional(),
  systolicBloodPressure: z.number().int().min(60).max(260).optional(),
  clinicalQuestion: z.string().trim().max(1000).optional(),
  imageName: z.string().trim().min(1).max(200),
  imageData: z.string().min(100).max(8_500_000),
});

export const reviewOutputSchema = z.object({
  id: z.number().int().positive(),
  severity: z.enum(["mild", "moderate", "severe", "proliferative"]),
  confidence: z.number().min(0).max(100),
  visualFindings: z.string().trim().min(10).max(3000),
  riskLevel: z.enum(["low", "elevated", "high"]),
  riskScore: z.number().min(0).max(100),
  groundedExplanation: z.string().trim().min(20).max(5000),
  evidence: z.array(z.object({
    sourceTitle: z.string().trim().min(3).max(255),
    sourceUrl: z.string().url().max(1024).optional(),
    publisher: z.string().trim().max(255).optional(),
    sourceType: z.enum(["guideline", "research", "institutional"]),
    excerpt: z.string().trim().min(20).max(4000),
    relevanceScore: z.number().min(0).max(100).optional(),
  })).min(1).max(8),
});

export const screeningRouter = router({
  list: protectedProcedure.query(({ ctx }) => listScreeningCasesForUser(ctx.user.id)),

  getById: protectedProcedure
    .input(z.object({ id: z.number().int().positive() }))
    .query(async ({ ctx, input }) => {
      const screeningCase = await getScreeningCaseForUser(input.id, ctx.user.id);
      if (!screeningCase) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Screening case not found." });
      }
      return screeningCase;
    }),

  create: protectedProcedure.input(screeningInputSchema).mutation(async ({ ctx, input }) => {
    const image = parseScreeningImage(input.imageData);
    if (!image) {
      throw new TRPCError({
        code: "BAD_REQUEST",
        message: "Upload a valid JPEG or PNG retinal image smaller than 6 MB.",
      });
    }

    const caseCode = buildCaseCode();
    const imageKey = `screenings/${ctx.user.id}/${caseCode}.${image.extension}`;
    const storedImage = await storagePut(imageKey, image.bytes, image.mimeType);

    return createScreeningCase({
      userId: ctx.user.id,
      caseCode,
      patientId: input.patientId,
      patientAge: input.patientAge,
      diabetesDurationYears: input.diabetesDurationYears,
      hba1c: input.hba1c,
      systolicBloodPressure: input.systolicBloodPressure,
      clinicalQuestion: input.clinicalQuestion,
      imageKey: storedImage.key,
      imageUrl: storedImage.url,
      imageName: input.imageName,
    });
  }),

  complete: protectedProcedure.input(reviewOutputSchema).mutation(async ({ ctx, input }) => {
    const completed = await saveScreeningReview({
      caseId: input.id,
      userId: ctx.user.id,
      severity: input.severity,
      confidence: input.confidence,
      visualFindings: input.visualFindings,
      riskLevel: input.riskLevel,
      riskScore: input.riskScore,
      groundedExplanation: input.groundedExplanation,
      evidence: input.evidence,
    });
    if (!completed) {
      throw new TRPCError({ code: "NOT_FOUND", message: "Screening case not found." });
    }
    return completed;
  }),
});
