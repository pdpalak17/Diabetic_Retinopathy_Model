import { decimal, index, int, mysqlEnum, mysqlTable, text, timestamp, uniqueIndex, varchar } from "drizzle-orm/mysql-core";

/** Core user table backing Manus OAuth. */
export const users = mysqlTable("users", {
  id: int("id").autoincrement().primaryKey(),
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

/** A clinician-created screening submission, including validated patient inputs and retinal image metadata. */
export const screeningCases = mysqlTable(
  "screeningCases",
  {
    id: int("id").autoincrement().primaryKey(),
    userId: int("userId").notNull().references(() => users.id, { onDelete: "cascade" }),
    caseCode: varchar("caseCode", { length: 32 }).notNull(),
    patientId: varchar("patientId", { length: 80 }).notNull(),
    patientAge: int("patientAge").notNull(),
    diabetesDurationYears: decimal("diabetesDurationYears", { precision: 5, scale: 1 }),
    hba1c: decimal("hba1c", { precision: 4, scale: 1 }),
    systolicBloodPressure: int("systolicBloodPressure"),
    clinicalQuestion: text("clinicalQuestion"),
    imageKey: varchar("imageKey", { length: 512 }).notNull(),
    imageUrl: varchar("imageUrl", { length: 1024 }).notNull(),
    imageName: varchar("imageName", { length: 255 }).notNull(),
    status: mysqlEnum("status", ["submitted", "reviewed"]).default("submitted").notNull(),
    createdAt: timestamp("createdAt").defaultNow().notNull(),
    updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  },
  table => [
    uniqueIndex("screeningCases_caseCode_unique").on(table.caseCode),
    index("screeningCases_user_updated_idx").on(table.userId, table.updatedAt),
  ],
);

/** Fused outputs from the visual, clinical-risk, retrieval, and explanation layers. */
export const screeningAssessments = mysqlTable(
  "screeningAssessments",
  {
    id: int("id").autoincrement().primaryKey(),
    caseId: int("caseId").notNull().references(() => screeningCases.id, { onDelete: "cascade" }),
    visualPipelineStatus: mysqlEnum("visualPipelineStatus", ["pending", "complete", "failed"]).default("pending").notNull(),
    severity: mysqlEnum("severity", ["unassessed", "mild", "moderate", "severe", "proliferative"]).default("unassessed").notNull(),
    confidence: decimal("confidence", { precision: 5, scale: 2 }),
    visualFindings: text("visualFindings"),
    clinicalPipelineStatus: mysqlEnum("clinicalPipelineStatus", ["pending", "complete", "failed"]).default("pending").notNull(),
    riskLevel: mysqlEnum("riskLevel", ["unassessed", "low", "elevated", "high"]).default("unassessed").notNull(),
    riskScore: decimal("riskScore", { precision: 5, scale: 2 }),
    evidencePipelineStatus: mysqlEnum("evidencePipelineStatus", ["pending", "complete", "failed"]).default("pending").notNull(),
    explanationPipelineStatus: mysqlEnum("explanationPipelineStatus", ["pending", "complete", "failed"]).default("pending").notNull(),
    groundedExplanation: text("groundedExplanation"),
    modelLimitations: text("modelLimitations"),
    createdAt: timestamp("createdAt").defaultNow().notNull(),
    updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  },
  table => [
    uniqueIndex("screeningAssessments_caseId_unique").on(table.caseId),
  ],
);

/** Source-attributed evidence returned by the retrieval pipeline for a screening case. */
export const screeningEvidence = mysqlTable(
  "screeningEvidence",
  {
    id: int("id").autoincrement().primaryKey(),
    caseId: int("caseId").notNull().references(() => screeningCases.id, { onDelete: "cascade" }),
    sourceTitle: varchar("sourceTitle", { length: 255 }).notNull(),
    sourceUrl: varchar("sourceUrl", { length: 1024 }),
    publisher: varchar("publisher", { length: 255 }),
    sourceType: mysqlEnum("sourceType", ["guideline", "research", "institutional"]).notNull(),
    excerpt: text("excerpt").notNull(),
    relevanceScore: decimal("relevanceScore", { precision: 5, scale: 2 }),
    createdAt: timestamp("createdAt").defaultNow().notNull(),
  },
  table => [index("screeningEvidence_case_idx").on(table.caseId)],
);

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;
export type ScreeningCase = typeof screeningCases.$inferSelect;
export type ScreeningAssessment = typeof screeningAssessments.$inferSelect;
export type ScreeningEvidence = typeof screeningEvidence.$inferSelect;
