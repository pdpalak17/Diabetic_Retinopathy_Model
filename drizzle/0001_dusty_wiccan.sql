CREATE TABLE `screeningAssessments` (
	`id` int AUTO_INCREMENT NOT NULL,
	`caseId` int NOT NULL,
	`visualPipelineStatus` enum('pending','complete','failed') NOT NULL DEFAULT 'pending',
	`severity` enum('unassessed','mild','moderate','severe','proliferative') NOT NULL DEFAULT 'unassessed',
	`confidence` decimal(5,2),
	`visualFindings` text,
	`clinicalPipelineStatus` enum('pending','complete','failed') NOT NULL DEFAULT 'pending',
	`riskLevel` enum('unassessed','low','elevated','high') NOT NULL DEFAULT 'unassessed',
	`riskScore` decimal(5,2),
	`evidencePipelineStatus` enum('pending','complete','failed') NOT NULL DEFAULT 'pending',
	`explanationPipelineStatus` enum('pending','complete','failed') NOT NULL DEFAULT 'pending',
	`groundedExplanation` text,
	`modelLimitations` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `screeningAssessments_id` PRIMARY KEY(`id`),
	CONSTRAINT `screeningAssessments_caseId_unique` UNIQUE(`caseId`)
);
--> statement-breakpoint
CREATE TABLE `screeningCases` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`caseCode` varchar(32) NOT NULL,
	`patientId` varchar(80) NOT NULL,
	`patientAge` int NOT NULL,
	`diabetesDurationYears` decimal(5,1),
	`hba1c` decimal(4,1),
	`systolicBloodPressure` int,
	`clinicalQuestion` text,
	`imageKey` varchar(512) NOT NULL,
	`imageUrl` varchar(1024) NOT NULL,
	`imageName` varchar(255) NOT NULL,
	`status` enum('submitted','reviewed') NOT NULL DEFAULT 'submitted',
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `screeningCases_id` PRIMARY KEY(`id`),
	CONSTRAINT `screeningCases_caseCode_unique` UNIQUE(`caseCode`)
);
--> statement-breakpoint
CREATE TABLE `screeningEvidence` (
	`id` int AUTO_INCREMENT NOT NULL,
	`caseId` int NOT NULL,
	`sourceTitle` varchar(255) NOT NULL,
	`sourceUrl` varchar(1024),
	`publisher` varchar(255),
	`sourceType` enum('guideline','research','institutional') NOT NULL,
	`excerpt` text NOT NULL,
	`relevanceScore` decimal(5,2),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `screeningEvidence_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
ALTER TABLE `screeningAssessments` ADD CONSTRAINT `screeningAssessments_caseId_screeningCases_id_fk` FOREIGN KEY (`caseId`) REFERENCES `screeningCases`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `screeningCases` ADD CONSTRAINT `screeningCases_userId_users_id_fk` FOREIGN KEY (`userId`) REFERENCES `users`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `screeningEvidence` ADD CONSTRAINT `screeningEvidence_caseId_screeningCases_id_fk` FOREIGN KEY (`caseId`) REFERENCES `screeningCases`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX `screeningCases_user_updated_idx` ON `screeningCases` (`userId`,`updatedAt`);--> statement-breakpoint
CREATE INDEX `screeningEvidence_case_idx` ON `screeningEvidence` (`caseId`);