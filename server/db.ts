import { and, desc, eq } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import {
  InsertUser,
  screeningAssessments,
  screeningCases,
  screeningEvidence,
  users,
} from "../drizzle/schema";
import { ENV } from "./_core/env";

let _db: ReturnType<typeof drizzle> | null = null;

export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) throw new Error("User openId is required for upsert");
  const db = await getDb();
  if (!db) return;

  const values: InsertUser = { openId: user.openId, lastSignedIn: user.lastSignedIn ?? new Date() };
  const updateSet: Record<string, unknown> = { lastSignedIn: values.lastSignedIn };
  (["name", "email", "loginMethod"] as const).forEach(field => {
    if (user[field] !== undefined) {
      values[field] = user[field] ?? null;
      updateSet[field] = user[field] ?? null;
    }
  });
  values.role = user.role ?? (user.openId === ENV.ownerOpenId ? "admin" : "user");
  updateSet.role = values.role;

  await db.insert(users).values(values).onDuplicateKeyUpdate({ set: updateSet });
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);
  return result[0];
}

type CreateScreeningCaseInput = {
  userId: number;
  caseCode: string;
  patientId: string;
  patientAge: number;
  diabetesDurationYears?: number;
  hba1c?: number;
  systolicBloodPressure?: number;
  clinicalQuestion?: string;
  imageKey: string;
  imageUrl: string;
  imageName: string;
};

export async function createScreeningCase(input: CreateScreeningCaseInput) {
  const db = await getDb();
  if (!db) throw new Error("Database connection is unavailable.");

  await db.insert(screeningCases).values({
    userId: input.userId,
    caseCode: input.caseCode,
    patientId: input.patientId,
    patientAge: input.patientAge,
    diabetesDurationYears: input.diabetesDurationYears === undefined ? null : String(input.diabetesDurationYears),
    hba1c: input.hba1c === undefined ? null : String(input.hba1c),
    systolicBloodPressure: input.systolicBloodPressure ?? null,
    clinicalQuestion: input.clinicalQuestion || null,
    imageKey: input.imageKey,
    imageUrl: input.imageUrl,
    imageName: input.imageName,
  });

  const [createdCase] = await db.select().from(screeningCases).where(eq(screeningCases.caseCode, input.caseCode)).limit(1);
  if (!createdCase) throw new Error("Screening case could not be created.");

  await db.insert(screeningAssessments).values({ caseId: createdCase.id });
  return getScreeningCaseForUser(createdCase.id, input.userId);
}

export async function listScreeningCasesForUser(userId: number) {
  const db = await getDb();
  if (!db) return [];

  return db
    .select({
      id: screeningCases.id,
      caseCode: screeningCases.caseCode,
      patientId: screeningCases.patientId,
      status: screeningCases.status,
      createdAt: screeningCases.createdAt,
      updatedAt: screeningCases.updatedAt,
      visualPipelineStatus: screeningAssessments.visualPipelineStatus,
      severity: screeningAssessments.severity,
      confidence: screeningAssessments.confidence,
      clinicalPipelineStatus: screeningAssessments.clinicalPipelineStatus,
      riskLevel: screeningAssessments.riskLevel,
    })
    .from(screeningCases)
    .leftJoin(screeningAssessments, eq(screeningAssessments.caseId, screeningCases.id))
    .where(eq(screeningCases.userId, userId))
    .orderBy(desc(screeningCases.updatedAt));
}

export async function getScreeningCaseForUser(caseId: number, userId: number) {
  const db = await getDb();
  if (!db) return null;

  const [record] = await db
    .select({ screeningCase: screeningCases, assessment: screeningAssessments })
    .from(screeningCases)
    .leftJoin(screeningAssessments, eq(screeningAssessments.caseId, screeningCases.id))
    .where(and(eq(screeningCases.id, caseId), eq(screeningCases.userId, userId)))
    .limit(1);
  if (!record) return null;

  const evidence = await db
    .select()
    .from(screeningEvidence)
    .where(eq(screeningEvidence.caseId, caseId))
    .orderBy(desc(screeningEvidence.relevanceScore));

  return { ...record, evidence };
}

type PersistedEvidence = {
  sourceTitle: string;
  sourceUrl?: string;
  publisher?: string;
  sourceType: "guideline" | "research" | "institutional";
  excerpt: string;
  relevanceScore?: number;
};

type SaveScreeningReviewInput = {
  caseId: number;
  userId: number;
  severity: "mild" | "moderate" | "severe" | "proliferative";
  confidence: number;
  visualFindings: string;
  riskLevel: "low" | "elevated" | "high";
  riskScore: number;
  groundedExplanation: string;
  evidence: PersistedEvidence[];
};

export async function saveScreeningReview(input: SaveScreeningReviewInput) {
  const db = await getDb();
  if (!db) throw new Error("Database connection is unavailable.");

  const existing = await getScreeningCaseForUser(input.caseId, input.userId);
  if (!existing) return null;

  await db.update(screeningAssessments).set({
    visualPipelineStatus: "complete",
    severity: input.severity,
    confidence: String(input.confidence),
    visualFindings: input.visualFindings,
    clinicalPipelineStatus: "complete",
    riskLevel: input.riskLevel,
    riskScore: String(input.riskScore),
    evidencePipelineStatus: "complete",
    explanationPipelineStatus: "complete",
    groundedExplanation: input.groundedExplanation,
    modelLimitations: "Recorded outputs support qualified professional review and do not constitute autonomous diagnosis or treatment advice.",
  }).where(eq(screeningAssessments.caseId, input.caseId));

  await db.delete(screeningEvidence).where(eq(screeningEvidence.caseId, input.caseId));
  if (input.evidence.length > 0) {
    await db.insert(screeningEvidence).values(input.evidence.map(item => ({
      caseId: input.caseId,
      sourceTitle: item.sourceTitle,
      sourceUrl: item.sourceUrl || null,
      publisher: item.publisher || null,
      sourceType: item.sourceType,
      excerpt: item.excerpt,
      relevanceScore: item.relevanceScore === undefined ? null : String(item.relevanceScore),
    })));
  }
  await db.update(screeningCases).set({ status: "reviewed" }).where(eq(screeningCases.id, input.caseId));
  return getScreeningCaseForUser(input.caseId, input.userId);
}
