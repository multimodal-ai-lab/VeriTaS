"""Database interface for the annotation tool."""
from veritas import database, logger
from veritas.db.base import Database
from veritas.common.annotation.rating import Rating


class AnnotationDB(Database):
    """Database for managing user annotations of claims."""

    async def connect_maybe_initialize(self):
        """Connects to the DB and initializes annotation tables if needed."""
        # Note: We assume the database already exists (created by VeritasDB)
        await self.connect()
        await self._create_tables()

    async def _create_tables(self):
        """Create annotation-specific tables."""

        # Users table
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS annotation_users (
                id SERIAL PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                code TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL,
                languages JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS annotation_users_code_idx ON annotation_users(code);
            """
        )

        # Annotations table
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id SERIAL PRIMARY KEY,
                user_id INT NOT NULL REFERENCES annotation_users(id) ON DELETE CASCADE,
                claim_id INT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
                status TEXT NOT NULL CHECK (status IN ('in_progress', 'completed', 'dismissed')),

                -- Dismissal status
                dismissed BOOLEAN DEFAULT FALSE,
                dismiss_reason TEXT,

                -- Clarity
                clarity TEXT,
                clarity_confidence INT CHECK (clarity_confidence BETWEEN 0 AND 3),
                clarity_uncertainty_rationale TEXT,
                clarity_interpretation_commitment TEXT,
                clarity_explanation TEXT,

                -- Context Coverage
                context_coverage TEXT,
                context_coverage_confidence INT CHECK (context_coverage_confidence BETWEEN 0 AND 3),
                context_coverage_uncertainty_rationale TEXT,
                context_coverage_explanation TEXT,

                -- Veracity
                veracity TEXT,
                veracity_tags JSONB,
                veracity_confidence INT CHECK (veracity_confidence BETWEEN 0 AND 3),
                veracity_uncertainty_rationale TEXT,
                veracity_explanation TEXT,

                -- Intent
                intent TEXT,
                intent_tags JSONB,
                intent_confidence INT CHECK (intent_confidence BETWEEN 0 AND 3),
                intent_uncertainty_rationale TEXT,

                -- Manual validation checks (True = passed, False = failed, NULL = not checked)
                manual_is_clear BOOLEAN DEFAULT TRUE,
                manual_media_no_annotations BOOLEAN DEFAULT TRUE,
                manual_text_no_verdict BOOLEAN DEFAULT TRUE,
                manual_all_media_present BOOLEAN DEFAULT TRUE,
                manual_validation_completed BOOLEAN DEFAULT FALSE,

                -- Metadata
                time_estimate TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                submitted_at TIMESTAMP,
                excluded BOOLEAN DEFAULT FALSE,
                meta_comment TEXT,

                -- Ensure each user annotates each claim only once
                CONSTRAINT unique_user_claim UNIQUE (user_id, claim_id)
            );
            CREATE INDEX IF NOT EXISTS annotations_user_id_idx ON annotations(user_id);
            CREATE INDEX IF NOT EXISTS annotations_claim_id_idx ON annotations(claim_id);
            CREATE INDEX IF NOT EXISTS annotations_status_idx ON annotations(status);
            """
        )

        # Media evaluations table
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS media_evaluations (
                id SERIAL PRIMARY KEY,
                annotation_id INT NOT NULL REFERENCES annotations(id) ON DELETE CASCADE,
                media_id INT NOT NULL,

                -- Dismissal status
                dismissed BOOLEAN DEFAULT FALSE,
                dismiss_reason TEXT,

                -- Media Authenticity
                media_authenticity TEXT,
                media_authenticity_tags JSONB,
                media_authenticity_confidence INT CHECK (media_authenticity_confidence BETWEEN 0 AND 3),
                media_authenticity_uncertainty_rationale TEXT,
                media_authenticity_explanation TEXT,

                -- Media Contextualization
                media_contextualization TEXT,
                media_contextualization_confidence INT CHECK (media_contextualization_confidence BETWEEN 0 AND 3),
                media_contextualization_uncertainty_rationale TEXT,
                media_contextualization_explanation TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- Ensure each media in an annotation is evaluated only once
                CONSTRAINT unique_annotation_media UNIQUE (annotation_id, media_id)
            );
            CREATE INDEX IF NOT EXISTS media_evaluations_annotation_id_idx ON media_evaluations(annotation_id);
            """
        )

        logger.info("Annotation tables created/verified.")

    # User operations
    async def insert_user(self, email: str, code: str, role: str, status: str = None) -> int:
        """Add a new annotation user and return the assigned ID."""
        if status is not None:
            query = """
            INSERT INTO annotation_users (email, code, role, status)
            VALUES ($1, $2, $3, $4)
            RETURNING id;
            """
            return await self._fetchval(query, email, code, role, status)
        else:
            query = """
                INSERT INTO annotation_users (email, code, role)
                VALUES ($1, $2, $3)
                RETURNING id;
            """
            return await self._fetchval(query, email, code, role)

    async def get_user_by_code(self, code: str) -> dict | None:
        """Retrieve a user by their code."""
        query = "SELECT * FROM annotation_users WHERE code = $1"
        row = await self._fetchrow(query, code)
        return dict(row) if row else None

    async def get_user_by_id(self, user_id: int) -> dict | None:
        """Retrieve a user by their ID."""
        query = "SELECT * FROM annotation_users WHERE id = $1"
        row = await self._fetchrow(query, user_id)
        return dict(row) if row else None

    async def update_user_languages(self, user_id: int, languages: list[str]) -> None:
        """Update user's language preferences."""
        query = "UPDATE annotation_users SET languages = $2 WHERE id = $1;"
        await self._execute(query, user_id, languages)

    async def get_user_languages(self, user_id: int) -> list[str]:
        """Get user's language preferences."""
        query = "SELECT languages FROM annotation_users WHERE id = $1"
        row = await self._fetchrow(query, user_id)
        return row['languages'] if row and row['languages'] else []

    # Annotation operations
    async def create_annotation(self, user_id: int, claim_id: int) -> int:
        """Create a new annotation and return its ID."""
        query = """
            INSERT INTO annotations (user_id, claim_id, status)
            VALUES ($1, $2, 'in_progress')
            RETURNING id;
        """
        return await self._fetchval(query, user_id, claim_id)

    async def get_annotation_by_id(self, annotation_id: int) -> dict | None:
        """Retrieve an annotation by ID."""
        query = "SELECT * FROM annotations WHERE id = $1"
        row = await self._fetchrow(query, annotation_id)
        return dict(row) if row else None

    async def get_in_progress_annotation(self, user_id: int) -> dict | None:
        """Get the user's current in-progress annotation if any."""
        query = """
            SELECT * FROM annotations
            WHERE user_id = $1 AND status = 'in_progress'
            LIMIT 1;
        """
        row = await self._fetchrow(query, user_id)
        return dict(row) if row else None

    async def update_annotation(self, annotation_id: int, fields: dict) -> None:
        """Update annotation fields. Automatically updates updated_at timestamp."""
        # Build dynamic UPDATE query
        set_clauses = [f"{key} = ${i+2}" for i, key in enumerate(fields.keys())]
        set_clause = ", ".join(set_clauses)

        query = f"""
            UPDATE annotations
            SET {set_clause}, updated_at = CURRENT_TIMESTAMP
            WHERE id = $1;
        """
        values = [annotation_id] + list(fields.values())
        await self._execute(query, *values)

    async def submit_annotation(self, annotation_id: int, time_estimate: str) -> None:
        """Mark annotation as completed with submission timestamp."""
        query = """
            UPDATE annotations
            SET status = 'completed',
                time_estimate = $2,
                submitted_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1;
        """
        await self._execute(query, annotation_id, time_estimate)

    async def dismiss_annotation(self, annotation_id: int, dismiss_reason: str) -> None:
        """Mark annotation as dismissed with reason."""
        query = """
            UPDATE annotations
            SET status = 'dismissed',
                dismissed = TRUE,
                dismiss_reason = $2,
                submitted_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1;
        """
        await self._execute(query, annotation_id, dismiss_reason)

    async def get_completed_annotations_by_user(self, user_id: int) -> list[dict]:
        """Get all completed annotations for a user."""
        query = """
            SELECT * FROM annotations
            WHERE user_id = $1 AND status = 'completed'
            ORDER BY submitted_at DESC;
        """
        rows = await self._fetch(query, user_id)
        return [dict(row) for row in rows]

    async def get_dismissed_annotations_by_user(self, user_id: int) -> list[dict]:
        """Get all dismissed annotations for a user."""
        query = """
            SELECT * FROM annotations
            WHERE user_id = $1 AND status = 'dismissed'
            ORDER BY submitted_at DESC;
        """
        rows = await self._fetch(query, user_id)
        return [dict(row) for row in rows]

    async def get_annotation_count_by_claim(self, claim_id: int) -> int:
        """Get the number of completed annotations for a specific claim."""
        query = """
            SELECT COUNT(*) FROM annotations
            WHERE claim_id = $1 AND status = 'completed';
        """
        return await self._fetchval(query, claim_id)

    async def get_annotation_counts_by_claims(self, claim_ids: list[int]) -> dict[int, int]:
        """Get the number of active annotations (in_progress + completed, non-excluded) for multiple claims."""
        if not claim_ids:
            return {}
        query = """
            SELECT claim_id, COUNT(*) as count
            FROM annotations
            WHERE claim_id = ANY($1) AND status IN ('in_progress', 'completed') AND NOT excluded
            GROUP BY claim_id;
        """
        rows = await self._fetch(query, claim_ids)
        result = {row['claim_id']: row['count'] for row in rows}
        # Add 0 for claims with no annotations
        for claim_id in claim_ids:
            if claim_id not in result:
                result[claim_id] = 0
        return result

    async def get_claims_annotated_by_user(self, user_id: int) -> list[int]:
        """Get list of claim IDs already annotated by this user."""
        query = """
            SELECT claim_id FROM annotations
            WHERE user_id = $1;
        """
        rows = await self._fetch(query, user_id)
        return [row['claim_id'] for row in rows]

    # Statistics operations
    async def get_annotation_distribution(self) -> dict[int, int]:
        """
        Get distribution of claims by number of annotators.

        Returns a dictionary mapping annotator_count -> claim_count.
        For example: {1: 50, 2: 30, 3: 15} means 50 claims have 1 annotator, 30 have 2, etc.
        Only counts completed annotations.
        """
        query = """
            SELECT annotation_count, COUNT(*) as claim_count
            FROM (
                SELECT claim_id, COUNT(*) as annotation_count
                FROM annotations
                WHERE status = 'completed' AND NOT excluded
                GROUP BY claim_id
            ) subq
            GROUP BY annotation_count
            ORDER BY annotation_count;
        """
        rows = await self._fetch(query)
        return {row['annotation_count']: row['claim_count'] for row in rows}

    async def get_claims_with_multiple_annotations(self, min_annotators: int = 2) -> list[dict]:
        """
        Get claims that have min_annotators or more completed annotations.

        Returns a list of dicts with claim_id, annotator_count, and annotation_ids.
        Ordered by annotator_count descending.
        """
        query = """
            SELECT
                claim_id,
                COUNT(*) as annotator_count,
                ARRAY_AGG(id ORDER BY id) as annotation_ids
            FROM annotations
            WHERE status = 'completed' AND NOT excluded
            GROUP BY claim_id
            HAVING COUNT(*) >= $1
            ORDER BY annotator_count DESC;
        """
        rows = await self._fetch(query, min_annotators)
        return [dict(row) for row in rows]

    async def get_annotations_for_agreement_analysis(self, claim_ids: list[int]) -> dict[int, list[dict]]:
        """
        Get all annotation data for specified claims, grouped by claim_id.

        Returns a dictionary mapping claim_id to a list of annotation dicts.
        Each annotation dict includes all evaluation fields needed for agreement analysis.
        Only includes completed annotations.
        """
        if not claim_ids:
            return {}

        # Get main annotation data
        query = """
            SELECT * FROM annotations
            WHERE claim_id = ANY($1) AND status = 'completed' AND NOT excluded
            ORDER BY claim_id, id;
        """
        rows = await self._fetch(query, claim_ids)

        # Group by claim_id
        result = {}
        for row in rows:
            claim_id = row['claim_id']
            if claim_id not in result:
                result[claim_id] = []
            result[claim_id].append(dict(row))

        return result

    async def get_media_evaluations_for_agreement_analysis(
        self, annotation_ids: list[int]
    ) -> dict[int, list[dict]]:
        """
        Get all media evaluations for specified annotations, grouped by annotation_id.

        Returns a dictionary mapping annotation_id to a list of media evaluation dicts.
        """
        if not annotation_ids:
            return {}

        query = """
            SELECT * FROM media_evaluations
            WHERE annotation_id = ANY($1)
            ORDER BY annotation_id, media_id;
        """
        rows = await self._fetch(query, annotation_ids)

        # Group by annotation_id
        result = {}
        for row in rows:
            annotation_id = row['annotation_id']
            if annotation_id not in result:
                result[annotation_id] = []
            result[annotation_id].append(dict(row))

        return result

    # Media evaluation operations
    async def insert_media_evaluation(
        self,
        annotation_id: int,
        media_id: int,
        fields: dict
    ) -> int:
        """Insert or update a media evaluation."""
        # Extract field names and values
        field_names = ['annotation_id', 'media_id'] + list(fields.keys())
        placeholders = [f'${i+1}' for i in range(len(field_names))]
        values = [annotation_id, media_id] + list(fields.values())

        # Build conflict update clause
        update_clauses = [f"{key} = EXCLUDED.{key}" for key in fields.keys()]
        update_clause = ", ".join(update_clauses)

        query = f"""
            INSERT INTO media_evaluations ({', '.join(field_names)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT (annotation_id, media_id)
            DO UPDATE SET {update_clause}, updated_at = CURRENT_TIMESTAMP
            RETURNING id;
        """
        return await self._fetchval(query, *values)

    async def get_media_evaluations_by_annotation(self, annotation_id: int) -> list[dict]:
        """Get all media evaluations for an annotation."""
        query = """
            SELECT * FROM media_evaluations
            WHERE annotation_id = $1
            ORDER BY media_id;
        """
        rows = await self._fetch(query, annotation_id)
        return [dict(row) for row in rows]

    async def get_media_evaluation(self, annotation_id: int, media_id: int) -> dict | None:
        """Get a specific media evaluation."""
        query = """
            SELECT * FROM media_evaluations
            WHERE annotation_id = $1 AND media_id = $2;
        """
        row = await self._fetchrow(query, annotation_id, media_id)
        return dict(row) if row else None

    async def get_most_common_veracity(self, claim_id: int) -> str | None:
        """Get the most common veracity value for a claim from completed annotations."""
        query = """
            SELECT veracity, COUNT(*) as count
            FROM annotations
            WHERE claim_id = $1 AND status = 'completed' AND veracity IS NOT NULL
            GROUP BY veracity
            ORDER BY count DESC, veracity
            LIMIT 1;
        """
        row = await self._fetchrow(query, claim_id)
        return row['veracity'] if row else None

    async def get_claim_languages(self, claim_ids: list[int]) -> dict[int, str | None]:
        """
        Get the language for each claim based on its reviews.

        Returns a dictionary mapping claim_id to language code.
        If a claim has multiple reviews with different languages, returns the first one.
        If a claim has no reviews or no language info, returns None.
        """
        if not claim_ids:
            return {}

        query = """
            SELECT
                c.id as claim_id,
                (
                    SELECT r.language
                    FROM reviews r
                    WHERE r.id = ANY(c.review_ids)
                    AND r.language IS NOT NULL
                    ORDER BY r.id
                    LIMIT 1
                ) as language
            FROM claims c
            WHERE c.id = ANY($1)
        """
        rows = await self._fetch(query, claim_ids)

        # Map claim_id to language (already filtered for first non-null)
        result = {row['claim_id']: row['language'] for row in rows}

        # Add None for claims without language info
        for claim_id in claim_ids:
            if claim_id not in result:
                result[claim_id] = None

        return result

    async def get_all_users(self) -> list[dict]:
        """Fetch all users from the annotation_users table."""
        query = "SELECT id, email, role FROM annotation_users"
        rows = await self._fetch(query)
        return [dict(r) for r in rows]

    async def get_dismissed_counts(self) -> dict[int, int]:
        """Fetch counts of dismissed annotations per user."""
        query = "SELECT user_id, COUNT(*) as count FROM annotations WHERE dismissed = TRUE GROUP BY user_id"
        rows = await self._fetch(query)
        return {r['user_id']: r['count'] for r in rows}

    async def get_completed_counts(self) -> dict[int, int]:
        """Fetch counts of completed, non-dismissed, non-excluded annotations per user."""
        query = """
            SELECT user_id, COUNT(*) as count
            FROM annotations
            WHERE status = 'completed' AND dismissed = FALSE AND NOT excluded
            GROUP BY user_id
        """
        rows = await self._fetch(query)
        return {r['user_id']: r['count'] for r in rows}

    async def get_excluded_counts(self) -> dict[int, int]:
        """Fetch counts of excluded annotations per user."""
        query = """
            SELECT user_id, COUNT(*) as count
            FROM annotations
            WHERE excluded
            GROUP BY user_id
        """
        rows = await self._fetch(query)
        return {r['user_id']: r['count'] for r in rows}

    async def get_completed_ratings(self) -> list[dict[str, int | dict[str, Rating]]]:
        """
        Fetch all claim-level ratings from completed, non-dismissed, and non-excluded annotations.

        Returns:
            A list of dicts, each with 'claim_id' and 'ratings' (a dict mapping property name to Rating).
        """
        # We need _value_to_sign and _human_to_rating logic here, but let's see if we can generalize it.
        # Actually, the issue description says "Extend the AnnotationsDB with methods that return the ratings as Rating objects".
        # I should probably move the conversion logic from plot_human_vs_automatic.py to here or a common place.

        query = """
            SELECT id as annotation_id, claim_id, user_id,
                   veracity, veracity_confidence, veracity_explanation,
                   context_coverage, context_coverage_confidence, context_coverage_explanation,
                   intent, intent_confidence, intent_uncertainty_rationale as intent_explanation
            FROM annotations
            WHERE status = 'completed' AND dismissed = FALSE AND NOT excluded
        """
        rows = await self._fetch(query)

        # To avoid circular imports or duplication, I'll implement a simple version of the conversion here
        # based on what's in plot_human_vs_automatic.py
        def _value_to_sign(prop: str, value: str | None) -> float:
            if not value: return 0.0
            v = str(value).strip().lower()
            if prop == "veracity": return +1.0 if v == "true" else (-1.0 if v == "false" else 0.0)
            if prop == "context_coverage": return +1.0 if v == "sufficient" else (-1.0 if v == "insufficient" else 0.0)
            if prop == "intent": return +1.0 if v == "legitimate" else (-1.0 if v == "illegitimate" else 0.0)
            return 0.0

        def _to_rating(prop: str, value: str | None, confidence: int | None, user_id: int, explanation: str | None = None) -> Rating | None:
            if value is None: return None
            sign = _value_to_sign(prop, value)
            magnitude = (confidence / 3.0) if confidence is not None else 0.0
            magnitude = max(0.0, min(1.0, magnitude))
            return Rating(score=sign * magnitude, rater=f"human:{user_id}", explanation=explanation)

        results = []
        for r in rows:
            ratings = {}
            for prop in ["veracity", "context_coverage", "intent"]:
                rating = _to_rating(prop, r[prop], r[f"{prop}_confidence"], r['user_id'], r[f"{prop}_explanation"])
                if rating:
                    ratings[prop] = rating
            results.append({"annotation_id": r['annotation_id'], "claim_id": r['claim_id'], "ratings": ratings})
        return results

    async def get_completed_media_ratings(self) -> list[dict[str, int | dict[str, Rating]]]:
        """
        Fetch all media-level ratings from completed, non-dismissed, and non-excluded annotations.

        Returns:
            A list of dicts, each with 'claim_id', 'media_id', and 'ratings' (a dict mapping property name to Rating).
        """
        query = """
            SELECT a.id as annotation_id, a.claim_id, me.media_id, a.user_id,
                   me.media_authenticity, me.media_authenticity_confidence, me.media_authenticity_explanation,
                   me.media_contextualization, me.media_contextualization_confidence, me.media_contextualization_explanation,
                   c.data as claim_data
            FROM media_evaluations me
            JOIN annotations a ON me.annotation_id = a.id
            JOIN claims c ON a.claim_id = c.id
            WHERE a.status = 'completed' AND a.dismissed = FALSE AND NOT a.excluded
              AND me.media_authenticity IS NOT NULL
        """
        rows = await self._fetch(query)

        def _value_to_sign(prop: str, value: str | None) -> float:
            if not value: return 0.0
            v = str(value).strip().lower()
            if prop == "authenticity": return +1.0 if v == "pristine" else (-1.0 if v == "fabricated" else 0.0)
            if prop == "contextualization": return +1.0 if v == "correct" else (-1.0 if v == "incorrect" else 0.0)
            return 0.0

        def _to_rating(prop: str, value: str | None, confidence: int | None, user_id: int, explanation: str | None = None) -> Rating | None:
            if value is None: return None
            sign = _value_to_sign(prop, value)
            magnitude = (confidence / 3.0) if confidence is not None else 0.0
            magnitude = max(0.0, min(1.0, magnitude))
            return Rating(score=sign * magnitude, rater=f"human:{user_id}", explanation=explanation)

        results = []
        for r in rows:
            ratings = {}
            for prop in ["authenticity", "contextualization"]:
                rating = _to_rating(prop, r[f"media_{prop}"], r[f"media_{prop}_confidence"], r['user_id'], r[f"media_{prop}_explanation"])
                if rating:
                    ratings[prop] = rating
            results.append({
                "annotation_id": r['annotation_id'],
                "claim_id": r['claim_id'],
                "media_id": r['media_id'],
                "claim_data": r['claim_data'],
                "ratings": ratings
            })
        return results


# Global instance
db_database, db_user, db_password, db_host, db_port = database.values()
annotation_db = AnnotationDB(
    user=db_user,
    password=db_password,
    host=db_host,
    port=db_port,
    database=db_database
)
