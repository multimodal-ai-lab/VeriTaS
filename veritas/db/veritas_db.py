from collections import Counter
from datetime import datetime, timezone, date

from ezmm import MultimodalSequence
from pydantic import BaseModel, HttpUrl

from veritas import database, logger
from veritas.common import Appearance, Claim, Review, VeritasBaseModel
from veritas.common.article import Article
from veritas.common.publisher import Publisher
from veritas.common.verdict import Verdict
from veritas.db.base import Database
from veritas.util import get_domain
from veritas.util.util import hash_int32

#: Matches an ezMM item reference such as `<image:42>` inside stored text. The ID
#: is length-capped so that the cast to a PostgreSQL `INTEGER` cannot overflow on
#: a bogus reference that happens to sit in some scraped page.
MEDIA_REF_PATTERN = r"<(image|video|audio):([0-9]{1,9})>"

#: SQL predicate on `media`: the medium occurs at least once somewhere in VeriTaS.
MEDIA_IS_USED = ("(CARDINALITY(claim_ids) > 0 OR CARDINALITY(appearance_ids) > 0 "
                 "OR CARDINALITY(article_ids) > 0 OR CARDINALITY(evidence_ids) > 0)")


def _usage_agg(source: str) -> str:
    """SQL collecting the distinct IDs of one source kind into an `INTEGER[]`."""
    return (f"COALESCE(ARRAY_AGG(DISTINCT source_id) FILTER (WHERE source = '{source}'), "
            f"'{{}}'::INTEGER[])")


class VeritasDB(Database):
    """Core VeriTaS data store, containing all pipeline data."""

    async def connect_maybe_initialize(self, max_connections: int = 1):
        """Connects to the DB and initializes it where needed."""
        await self._ensure_db()
        await self.connect(max_connections)
        await self._create_tables()

    async def _create_tables(self):
        # Reviews
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS reviews
            (
                id                       SERIAL PRIMARY KEY,
                url                      TEXT                NOT NULL,
                google_claim_review      JSONB,
                datacommons_claim_review JSONB,
                direct_claim_review      JSONB,
                raw_claim                TEXT                NOT NULL,
                raw_claim_hash           INTEGER             NOT NULL,
                raw_rating               TEXT,
                raw_claimant_name        TEXT,
                raw_claimant_url         TEXT,
                raw_claim_date           TIMESTAMP,
                raw_publisher_name       TEXT,
                raw_publisher_url        TEXT,
                published                TIMESTAMP,
                author_name              TEXT,
                author_url               TEXT,
                modified                 TIMESTAMP,
                language                 TEXT,
                stage                    INT       DEFAULT 0 NOT NULL,
                dismissed                BOOLEAN   DEFAULT FALSE,
                dismissed_reason         TEXT      DEFAULT NULL,
                claim_id                 INT,
                publisher_id             INT,
                appearance_ids           INTEGER[],
                deferred_until           TIMESTAMP,
                created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                -- Unique constraint to avoid duplicates
                CONSTRAINT unique_reviews
                    UNIQUE (url, raw_claim_hash)
            );
            CREATE INDEX IF NOT EXISTS reviews_url_idx ON reviews (url);
            -- Optional helper index for checking deferral windows on reviews
            CREATE INDEX IF NOT EXISTS reviews_deferred_until_idx
                ON reviews (deferred_until);
            """
        )

        # Publishers (of Reviews and Articles)
        query = """
                CREATE TABLE IF NOT EXISTS publishers
                (
                    id            SERIAL PRIMARY KEY,
                    name          TEXT UNIQUE   NOT NULL,
                    domains       TEXT[] UNIQUE NOT NULL,
                    ifcn_status   VARCHAR(32),
                    ifcn_expires  TIMESTAMP,
                    efcsn_status  VARCHAR(32),
                    efcsn_expires TIMESTAMP,
                    country       TEXT,
                    language      VARCHAR(3),
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS publishers_domains_idx
                    ON publishers
                        USING GIN (domains);

                -- Performance index for publisher name filtering (case-insensitive search)
                -- Note: Requires pg_trgm extension for LIKE queries
                -- To enable: CREATE EXTENSION IF NOT EXISTS pg_trgm;
                CREATE INDEX IF NOT EXISTS publishers_name_lower_idx
                    ON publishers (LOWER(name));
                """
        await self._execute(query)

        # Articles
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS articles
            (
                id                SERIAL PRIMARY KEY,
                url               TEXT      NOT NULL UNIQUE,
                publisher_id      INT       NOT NULL,
                review_ids        INTEGER[] NOT NULL,
                scraped_page      TEXT,
                extracted_article TEXT,
                title             TEXT,
                dismissed         BOOLEAN   DEFAULT FALSE,
                dismissed_reason  TEXT      DEFAULT NULL,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS articles_url_idx ON articles (url);
            """
        )

        # Appearances
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS appearances
            (
                id                       SERIAL PRIMARY KEY,
                url                      TEXT UNIQUE,
                archive_url              TEXT UNIQUE,
                published                TIMESTAMP,
                author_name              TEXT,
                author_url               TEXT,
                original_scraped_content TEXT,
                archived_scraped_content TEXT,
                original_scrape_ok       BOOLEAN   DEFAULT FALSE,
                archived_scrape_ok       BOOLEAN   DEFAULT FALSE,
                scrape_method            TEXT,
                dismissed                BOOLEAN   DEFAULT FALSE,
                dismissed_reason         TEXT      DEFAULT NULL,
                deferred_until           TIMESTAMP,
                created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS appearances_url_unique_not_null
                ON appearances (url)
                WHERE url IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS appearances_archive_url_unique_not_null
                ON appearances (archive_url)
                WHERE archive_url IS NOT NULL;
            -- Optional helper index for checking deferral windows
            CREATE INDEX IF NOT EXISTS appearances_deferred_until_idx
                ON appearances (deferred_until);
            """
        )

        # Claims
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS claims
            (
                id                       SERIAL PRIMARY KEY,
                data                     TEXT      NOT NULL,
                date                     TIMESTAMP,
                language                 TEXT,
                appearance_ids           INTEGER[] NOT NULL,
                review_ids               INTEGER[] NOT NULL,
                verdict_ids              INTEGER[],
                dismissed                BOOLEAN   DEFAULT FALSE,
                dismissed_reason         TEXT      DEFAULT NULL,
                rectifiable              BOOLEAN   DEFAULT NULL,
                is_ambiguous             BOOLEAN   DEFAULT NULL,
                is_inconsistent          BOOLEAN   DEFAULT NULL,
                is_unshareable           BOOLEAN   DEFAULT NULL,
                media_expose_verdict     BOOLEAN   DEFAULT NULL,
                text_exposes_verdict     BOOLEAN   DEFAULT NULL,
                missing_referenced_media BOOLEAN   DEFAULT NULL,
                check_completed          BOOLEAN   DEFAULT NULL,
                media_origin             TEXT      DEFAULT NULL,
                is_rectified             BOOLEAN   DEFAULT FALSE,
                variant_id               INTEGER,
                released_quarter         BOOLEAN   DEFAULT FALSE,
                released_longitudinal    BOOLEAN   DEFAULT FALSE,
                text_embedding           FLOAT[],
                created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            -- Ensure there is only one original claim per combination of reviews
            CREATE UNIQUE INDEX IF NOT EXISTS unique_original_claim
                ON claims (review_ids)
                WHERE is_rectified IS FALSE;
            -- Ensure there is only one rectification per claim
            CREATE UNIQUE INDEX IF NOT EXISTS unique_rectification
                ON claims (variant_id)
                WHERE is_rectified IS TRUE;

            -- Performance indexes for filtering
            CREATE INDEX IF NOT EXISTS claims_date_idx ON claims (date);
            CREATE INDEX IF NOT EXISTS claims_dismissed_idx ON claims (dismissed);
            CREATE INDEX IF NOT EXISTS claims_is_rectified_idx ON claims (is_rectified);
            CREATE INDEX IF NOT EXISTS claims_rectifiable_idx ON claims (rectifiable);
            """
        )

        # Gold Evidence Reconstruction: additive-only claim columns.
        # These are written exclusively by `veritas.gold_evidence`; existing claim
        # data (incl. `dismissed`) is never touched by that pipeline.
        await self._execute(
            """
            ALTER TABLE claims ADD COLUMN IF NOT EXISTS gold_evidence_status     TEXT;
            ALTER TABLE claims ADD COLUMN IF NOT EXISTS gold_evidence_reason     TEXT;
            ALTER TABLE claims ADD COLUMN IF NOT EXISTS gold_evidence_updated_at TIMESTAMP;
            CREATE INDEX IF NOT EXISTS claims_gold_evidence_status_idx
                ON claims (gold_evidence_status);
            """
        )

        # Media. Formerly `media_embeddings`: the table now indexes every medium
        # VeriTaS refers to and records where it is used, with the embedding being
        # one (optional) column among them.
        #
        # Existing embeddings are carried over in every case and never recomputed:
        # the table is *renamed*, so the rows stay exactly where they are, and if a
        # `media` table already exists the embeddings are copied into it instead.
        # Nothing below drops a table or writes to `embedding`.
        await self._execute(
            """
            DO
            $$
                BEGIN
                    IF to_regclass('public.media_embeddings') IS NOT NULL
                        AND to_regclass('public.media') IS NULL THEN
                        ALTER TABLE media_embeddings RENAME TO media;
                    END IF;
                END
            $$;

            CREATE TABLE IF NOT EXISTS media
            (
                id             SERIAL PRIMARY KEY,
                kind           TEXT      NOT NULL,
                media_id       INTEGER   NOT NULL,
                embedding      FLOAT[],
                -- Where the medium occurs. Empty everywhere <=> unused by VeriTaS.
                claim_ids      INTEGER[] NOT NULL DEFAULT '{}',
                appearance_ids INTEGER[] NOT NULL DEFAULT '{}',
                article_ids    INTEGER[] NOT NULL DEFAULT '{}',
                evidence_ids   INTEGER[] NOT NULL DEFAULT '{}',
                indexed_at     TIMESTAMP,
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (kind, media_id)
            );

            -- Additive upgrade of a table that came from `media_embeddings`: it has
            -- neither the usage columns nor a nullable embedding (a medium without
            -- an embedding must be indexable, too).
            ALTER TABLE media ALTER COLUMN embedding DROP NOT NULL;
            ALTER TABLE media ADD COLUMN IF NOT EXISTS claim_ids      INTEGER[] NOT NULL DEFAULT '{}';
            ALTER TABLE media ADD COLUMN IF NOT EXISTS appearance_ids INTEGER[] NOT NULL DEFAULT '{}';
            ALTER TABLE media ADD COLUMN IF NOT EXISTS article_ids    INTEGER[] NOT NULL DEFAULT '{}';
            ALTER TABLE media ADD COLUMN IF NOT EXISTS evidence_ids   INTEGER[] NOT NULL DEFAULT '{}';
            ALTER TABLE media ADD COLUMN IF NOT EXISTS indexed_at     TIMESTAMP;

            CREATE INDEX IF NOT EXISTS media_kind_idx ON media (kind);
            CREATE INDEX IF NOT EXISTS media_claim_ids_idx
                ON media USING GIN (claim_ids);
            CREATE INDEX IF NOT EXISTS media_appearance_ids_idx
                ON media USING GIN (appearance_ids);
            CREATE INDEX IF NOT EXISTS media_article_ids_idx
                ON media USING GIN (article_ids);
            CREATE INDEX IF NOT EXISTS media_evidence_ids_idx
                ON media USING GIN (evidence_ids);

            DO
            $$
                BEGIN
                    -- Reached only if `media_embeddings` survived the rename above,
                    -- i.e. if a `media` table already existed. Its embeddings would
                    -- otherwise be stranded in a table nothing reads any more, so
                    -- copy them over - without overwriting anything `media` already
                    -- has, and without dropping `media_embeddings`.
                    IF to_regclass('public.media_embeddings') IS NOT NULL THEN
                        EXECUTE $migrate$
                            INSERT INTO media (kind, media_id, embedding)
                            SELECT kind, media_id, embedding
                            FROM media_embeddings
                            ON CONFLICT (kind, media_id) DO UPDATE
                            SET embedding = COALESCE(media.embedding, EXCLUDED.embedding);
                        $migrate$;
                    END IF;
                END
            $$;
            """
        )

        # Verdicts
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS verdicts
            (
                id                           SERIAL PRIMARY KEY,
                claim_id                     INT REFERENCES claims (id) ON DELETE CASCADE,
                review_ids                   INTEGER[] NOT NULL,
                media                        JSONB,
                veracity                     FLOAT,
                veracity_explanation         JSONB,
                context_coverage             FLOAT,
                context_coverage_explanation JSONB,
                integrity                    FLOAT,
                full_verdict                 JSONB,
                is_current                   BOOLEAN   DEFAULT TRUE,
                created_at                   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            -- Ensure there is only one current verdict at a time per claim
            CREATE UNIQUE INDEX IF NOT EXISTS unique_current_verdicts
                ON verdicts (claim_id)
                WHERE is_current IS TRUE;

            -- Performance indexes for verdict filtering
            CREATE INDEX IF NOT EXISTS verdicts_claim_id_current_idx
                ON verdicts (claim_id, is_current);
            CREATE INDEX IF NOT EXISTS verdicts_veracity_idx
                ON verdicts (veracity) WHERE is_current = TRUE;
            CREATE INDEX IF NOT EXISTS verdicts_integrity_idx
                ON verdicts (integrity) WHERE is_current = TRUE;
            CREATE INDEX IF NOT EXISTS verdicts_context_coverage_idx
                ON verdicts (context_coverage) WHERE is_current = TRUE;

            -- GIN index for JSONB media column (supports media verdict filtering)
            CREATE INDEX IF NOT EXISTS verdicts_media_gin_idx
                ON verdicts USING GIN (media) WHERE is_current = TRUE;
            """
        )

        # Reconstructed gold evidence
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS evidence
            (
                id                      SERIAL PRIMARY KEY,
                claim_id                INT     NOT NULL REFERENCES claims (id) ON DELETE CASCADE,
                review_id               INT,
                article_id              INT,
                proposition             TEXT    NOT NULL,
                proposition_hash        INTEGER NOT NULL,
                source_name             TEXT,
                source_kind             TEXT,
                -- NULL for sources that are not publications: a tool without a
                -- page, or offline evidence such as an interview.
                source_locator          TEXT,
                source_locator_hash     INTEGER,
                source_proximity        TEXT,
                source_raw_content      TEXT,
                available_since         TIMESTAMP,
                role                    TEXT,
                accessed_at             TIMESTAMP,
                extraction_reasoning    TEXT,
                extraction_confidence   FLOAT,
                accessible              BOOLEAN,
                faithfulness_assessment FLOAT,
                faithfulness_reasoning  TEXT,
                faithfulness_justification TEXT,
                before_fact_check       BOOLEAN,
                before_claim            BOOLEAN,
                later_event             BOOLEAN,
                temporal_reasoning      TEXT,
                temporal_justification  TEXT,
                admissible              BOOLEAN,
                inadmissibility_reason  TEXT,
                dismissed               BOOLEAN   DEFAULT FALSE,
                dismissed_reason        TEXT      DEFAULT NULL,
                deferred_until          TIMESTAMP,
                full_evidence           JSONB,
                created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_evidence
                    UNIQUE (claim_id, source_locator_hash, proposition_hash)
            );
            CREATE INDEX IF NOT EXISTS evidence_claim_id_idx
                ON evidence (claim_id);
            CREATE INDEX IF NOT EXISTS evidence_admissible_idx
                ON evidence (claim_id, admissible);
            CREATE INDEX IF NOT EXISTS evidence_available_since_idx
                ON evidence (available_since);

            -- Optional helper index for checking deferral windows on evidence
            CREATE INDEX IF NOT EXISTS evidence_deferred_until_idx
                ON evidence (deferred_until);

            -- Migrations for tables created by an earlier version
            ALTER TABLE evidence ADD COLUMN IF NOT EXISTS deferred_until TIMESTAMP;
            ALTER TABLE evidence DROP COLUMN IF EXISTS professional_fact_check;
            ALTER TABLE evidence DROP COLUMN IF EXISTS concurrent_fact_check;
            ALTER TABLE evidence ALTER COLUMN source_locator DROP NOT NULL;
            ALTER TABLE evidence ALTER COLUMN source_locator_hash DROP NOT NULL;
            """
        )

        # A missing locator must not defeat the uniqueness constraint: by default
        # PostgreSQL treats NULLs as distinct, so two extractions of the same
        # offline source would insert twice instead of upserting. NULLS NOT
        # DISTINCT fixes that from PostgreSQL 15 on; older servers keep the plain
        # constraint, where locator-less rows are simply not covered by it.
        await self._execute(
            """
            DO
            $$
                DECLARE
                    nulls_are_distinct BOOLEAN;
                BEGIN
                    IF current_setting('server_version_num')::INT < 150000 THEN
                        RETURN;
                    END IF;
                    EXECUTE 'SELECT EXISTS (SELECT 1
                                            FROM pg_constraint c
                                                     JOIN pg_index i ON i.indexrelid = c.conindid
                                            WHERE c.conrelid = ''evidence''::REGCLASS
                                              AND c.conname = ''unique_evidence''
                                              AND NOT i.indnullsnotdistinct)'
                        INTO nulls_are_distinct;
                    IF nulls_are_distinct THEN
                        ALTER TABLE evidence DROP CONSTRAINT unique_evidence;
                        EXECUTE 'ALTER TABLE evidence
                            ADD CONSTRAINT unique_evidence
                                UNIQUE NULLS NOT DISTINCT (claim_id, source_locator_hash,
                                                           proposition_hash)';
                    END IF;
                END
            $$;
            """
        )

        # Per-claim, per-condition outcome of the sufficiency validator
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS gold_evidence_results
            (
                id                SERIAL PRIMARY KEY,
                claim_id          INT  NOT NULL REFERENCES claims (id) ON DELETE CASCADE,
                condition         TEXT NOT NULL,
                ensemble_mode     TEXT NOT NULL,
                n_evidence        INT       DEFAULT 0,
                predicted_verdict JSONB,
                member_responses  JSONB,
                property_diffs    JSONB,
                max_property_diff FLOAT,
                is_close          BOOLEAN,
                threshold         FLOAT,
                model_specifiers  TEXT[],
                error             TEXT,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_gold_evidence_result
                    UNIQUE (claim_id, condition, ensemble_mode)
            );
            CREATE INDEX IF NOT EXISTS gold_evidence_results_claim_idx
                ON gold_evidence_results (claim_id);
            """
        )

    async def insert_review(self, review: Review) -> int:
        """Adds the review to the database and returns the assigned ID."""
        query = """
                INSERT INTO reviews (url, google_claim_review, datacommons_claim_review, direct_claim_review,
                                     raw_claim, raw_claim_hash, raw_rating, raw_claimant_name, raw_claimant_url,
                                     raw_claim_date, raw_publisher_name, raw_publisher_url, published,
                                     author_name, author_url, modified, language, stage, dismissed,
                                     dismissed_reason, claim_id, publisher_id, appearance_ids, deferred_until)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                        $22, $23, $24)
                RETURNING id; \
                """
        identifier = await self._fetchval(
            query,
            str(review.url),
            review.google_claim_review,
            review.datacommons_claim_review,
            review.direct_claim_review,
            review.raw_claim,
            hash_int32(review.raw_claim),
            review.raw_rating,
            review.raw_claimant_name,
            str(review.raw_claimant_url) if review.raw_claimant_url else None,
            review.raw_claim_date,
            review.raw_publisher_name,
            str(review.raw_publisher_url) if review.raw_publisher_url else None,
            review.published,
            review.author_name,
            str(review.author_url) if review.author_url else None,
            review.modified,
            review.language,
            review.stage,
            review.dismissed,
            review.dismissed_reason,
            review.claim_id,
            review.publisher_id,
            review.appearance_ids,
            review.deferred_until,
        )
        return identifier

    async def update_review(self, review: Review):
        """Updates an existing review in the database. URL cannot be changed."""
        query = """
                UPDATE reviews
                SET google_claim_review      = $1,
                    datacommons_claim_review = $2,
                    direct_claim_review      = $3,
                    raw_claim                = $4,
                    raw_claim_hash           = $5,
                    raw_rating               = $6,
                    raw_claimant_name        = $7,
                    raw_claimant_url         = $8,
                    raw_claim_date           = $9,
                    raw_publisher_name       = $10,
                    raw_publisher_url        = $11,
                    published                = $12,
                    author_name              = $13,
                    author_url               = $14,
                    modified                 = $15,
                    language                 = $16,
                    stage                    = $17,
                    dismissed                = $18,
                    dismissed_reason         = $19,
                    claim_id                 = $20,
                    publisher_id             = $21,
                    appearance_ids           = $22,
                    deferred_until           = $23,
                    updated_at               = CURRENT_TIMESTAMP
                WHERE id = $24; \
                """

        def _normalize_dt(dt):
            if isinstance(dt, datetime):
                if dt.tzinfo is not None:
                    # Convert to UTC and make it naive
                    return dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt

        await self._execute(
            query,
            review.google_claim_review,
            review.datacommons_claim_review,
            review.direct_claim_review,
            review.raw_claim,
            hash_int32(review.raw_claim),
            review.raw_rating,
            review.raw_claimant_name,
            str(review.raw_claimant_url) if review.raw_claimant_url else None,
            _normalize_dt(review.raw_claim_date),
            review.raw_publisher_name,
            str(review.raw_publisher_url) if review.raw_publisher_url else None,
            _normalize_dt(review.published),
            review.author_name,
            str(review.author_url) if review.author_url else None,
            _normalize_dt(review.modified),
            review.language,
            review.stage,
            review.dismissed,
            review.dismissed_reason,
            review.claim_id,
            review.publisher_id,
            review.appearance_ids,
            review.deferred_until,
            review.id,
        )

    async def save_claim_review(self, claim_reviews: dict[tuple[str, str], dict], source: str):
        """Inserts the raw ClaimReview dicts, identified by their URL-claim-pair,
        to the database. Updates the 'updated_at' field of the corresponding
        review *only when* the insert made a change. `source` is either 'google'
        or 'datacommons'."""
        url_claim_pairs = list(claim_reviews.keys())
        input_crs = claim_reviews.values()
        existing_crs = await self.get_claim_reviews(url_claim_pairs, source)
        assert len(input_crs) == len(existing_crs)

        # Compare input and existing CRs to identify new ones (those that differ from the existing)
        is_new = [cr_in != cr_existing for cr_in, cr_existing in zip(input_crs, existing_crs)]

        if any(is_new):
            # Filter for ClaimReviews that are actually new (or differ from existing ones)
            to_insert = [
                (url, claim, hash_int32(claim), cr)
                for (url, claim), cr, new in zip(url_claim_pairs, input_crs, is_new)
                if new
            ]

            # Bulk-insert the ClaimReviews
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 1. Create a temp table for staging
                    await conn.execute(
                        f"""
                        CREATE TEMP TABLE temp_claim_reviews (
                            url TEXT,
                            raw_claim TEXT,
                            raw_claim_hash INTEGER,
                            {source}_claim_review JSONB
                        ) ON COMMIT DROP;
                    """
                    )

                    # 2. Copy data into the temp table
                    await conn.copy_records_to_table(
                        table_name="temp_claim_reviews",
                        records=to_insert,
                        columns=["url", "raw_claim", "raw_claim_hash", f"{source}_claim_review"],
                    )

                    # 3. Insert, but update existing reviews
                    await conn.execute(
                        f"""
                        INSERT INTO reviews (url, raw_claim, raw_claim_hash, {source}_claim_review)
                        SELECT url, raw_claim, raw_claim_hash, {source}_claim_review FROM temp_claim_reviews
                        ON CONFLICT (url, raw_claim_hash) DO UPDATE
                        SET {source}_claim_review = EXCLUDED.{source}_claim_review,
                            updated_at = CURRENT_TIMESTAMP;
                    """
                    )

    async def insert_publisher(self, publisher: Publisher) -> int:
        """Adds the publisher to the database and returns the assigned ID.
        Updates the publisher (except for the URL) if it already exists."""
        query = """
                INSERT INTO publishers (name, domains, ifcn_status, ifcn_expires, efcsn_status, efcsn_expires, country,
                                        language)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id;
                """
        identifier = await self._fetchval(
            query,
            publisher.name,
            publisher.domains,
            publisher.ifcn_status,
            publisher.ifcn_expires,
            publisher.efcsn_status,
            publisher.efcsn_expires,
            publisher.country,
            publisher.language,
        )
        return identifier

    async def update_publisher(self, publisher: Publisher):
        """Updates an existing publisher in the database."""
        query = """
                UPDATE publishers
                SET name          = $1,
                    domains       = $2,
                    ifcn_status   = $3,
                    ifcn_expires  = $4,
                    efcsn_status  = $5,
                    efcsn_expires = $6,
                    country       = $7,
                    language      = $8,
                    updated_at    = CURRENT_TIMESTAMP
                WHERE id = $9; \
                """
        await self._execute(
            query,
            publisher.name,
            publisher.domains,
            publisher.ifcn_status,
            publisher.ifcn_expires,
            publisher.efcsn_status,
            publisher.efcsn_expires,
            publisher.country,
            publisher.language,
            publisher.id,
        )

    async def insert_article(self, article: Article) -> int:
        """Adds the article to the database and returns the assigned ID."""
        query = """
                INSERT INTO articles (url, publisher_id, review_ids, scraped_page, extracted_article, title, dismissed,
                                      dismissed_reason)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id; \
                """
        identifier = await self._fetchval(
            query,
            str(article.url),
            article.publisher_id,
            article.review_ids,
            article.scraped_page,
            article.extracted_article,
            article.title,
            article.dismissed,
            article.dismissed_reason,
        )
        return identifier

    async def update_article(self, article: Article):
        """Updates an existing article in the database."""
        query = """
                UPDATE articles
                SET url               = $1,
                    publisher_id      = $2,
                    review_ids        = $3,
                    scraped_page      = $4,
                    extracted_article = $5,
                    title             = $6,
                    dismissed         = $7,
                    dismissed_reason  = $8,
                    updated_at        = CURRENT_TIMESTAMP
                WHERE id = $9; \
                """
        await self._execute(
            query,
            str(article.url),
            article.publisher_id,
            article.review_ids,
            article.scraped_page,
            article.extracted_article,
            article.title,
            article.dismissed,
            article.dismissed_reason,
            article.id,
        )

    async def insert_appearance(self, appearance: Appearance) -> int:
        """Adds the appearance to the database and returns the assigned ID."""
        query = """
                INSERT INTO appearances (url, archive_url, published, author_name, author_url,
                                         original_scraped_content, archived_scraped_content,
                                         original_scrape_ok, archived_scrape_ok,
                                         scrape_method, dismissed, dismissed_reason, deferred_until)
                VALUES ($1, $2, $3, $4, $5,
                        $6, $7,
                        $8, $9,
                        $10, $11, $12, $13)
                RETURNING id; \
                """
        identifier = await self._fetchval(
            query,
            str(appearance.url) if appearance.url else None,
            str(appearance.archive_url) if appearance.archive_url else None,
            appearance.published,
            appearance.author_name,
            str(appearance.author_url) if appearance.author_url else None,
            str(appearance.original_scraped_content) if appearance.original_scraped_content else None,
            str(appearance.archived_scraped_content) if appearance.archived_scraped_content else None,
            appearance.original_scrape_ok,
            appearance.archived_scrape_ok,
            appearance.scrape_method,
            appearance.dismissed,
            appearance.dismissed_reason,
            appearance.deferred_until,
        )
        return identifier

    async def update_appearance(self, appearance: Appearance):
        """Updates an existing appearance in the database."""
        query = """
                UPDATE appearances
                SET url                      = $1,
                    archive_url              = $2,
                    published                = $3,
                    author_name              = $4,
                    author_url               = $5,
                    original_scraped_content = $6,
                    archived_scraped_content = $7,
                    original_scrape_ok       = $8,
                    archived_scrape_ok       = $9,
                    scrape_method            = $10,
                    dismissed                = $11,
                    dismissed_reason         = $12,
                    deferred_until           = $13,
                    updated_at               = CURRENT_TIMESTAMP
                WHERE id = $14; \
                """
        await self._execute(
            query,
            str(appearance.url) if appearance.url else None,
            str(appearance.archive_url) if appearance.archive_url else None,
            appearance.published,
            appearance.author_name,
            str(appearance.author_url) if appearance.author_url else None,
            str(appearance.original_scraped_content) if appearance.original_scraped_content else None,
            str(appearance.archived_scraped_content) if appearance.archived_scraped_content else None,
            appearance.original_scrape_ok,
            appearance.archived_scrape_ok,
            appearance.scrape_method,
            appearance.dismissed,
            appearance.dismissed_reason,
            appearance.deferred_until,
            appearance.id,
        )

    async def insert_claim(self, claim: Claim) -> int:
        """Adds the claim to the database and returns the assigned ID."""
        query = """
                INSERT INTO claims (data, date, language, appearance_ids, review_ids, verdict_ids,
                                    dismissed, dismissed_reason, rectifiable,
                                    is_ambiguous, is_inconsistent, is_unshareable,
                                    media_expose_verdict, text_exposes_verdict, missing_referenced_media,
                                    check_completed, media_origin,
                                    is_rectified, variant_id, text_embedding,
                                    gold_evidence_status, gold_evidence_reason,
                                    gold_evidence_updated_at)
                VALUES ($1, $2, $3, $4, $5, $6,
                        $7, $8, $9,
                        $10, $11, $12,
                        $13, $14, $15,
                        $16, $17,
                        $18, $19, $20,
                        $21, $22, $23)
                RETURNING id;
                """
        identifier = await self._fetchval(
            query,
            claim.data,
            claim.date,
            claim.language,
            claim.appearance_ids,
            claim.review_ids,
            claim.verdict_ids,
            claim.dismissed,
            claim.dismissed_reason,
            claim.rectifiable,
            claim.is_ambiguous,
            claim.is_inconsistent,
            claim.is_unshareable,
            claim.media_expose_verdict,
            claim.text_exposes_verdict,
            claim.missing_referenced_media,
            claim.check_completed,
            claim.media_origin,
            claim.is_rectified,
            claim.variant_id,
            claim.text_embedding,
            claim.gold_evidence_status,
            claim.gold_evidence_reason,
            claim.gold_evidence_updated_at,
        )
        return identifier

    async def insert_media_embedding(self, kind: str, media_id: int, embedding: list[float]):
        """Inserts a media embedding into the database."""
        query = """
                INSERT INTO media (kind, media_id, embedding)
                VALUES ($1, $2, $3)
                ON CONFLICT (kind, media_id) DO UPDATE
                SET embedding  = $3,
                    created_at = CURRENT_TIMESTAMP;
                """
        await self._execute(query, kind, media_id, embedding)

    async def update_claim(self, claim: Claim):
        """Updates an existing claim in the database."""
        query = """
                UPDATE claims
                SET data                     = $1,
                    date                     = $2,
                    language                 = $3,
                    appearance_ids           = $4,
                    review_ids               = $5,
                    verdict_ids              = $6,
                    dismissed                = $7,
                    dismissed_reason         = $8,
                    rectifiable              = $9,
                    is_ambiguous             = $10,
                    is_inconsistent          = $11,
                    is_unshareable           = $12,
                    media_expose_verdict     = $13,
                    text_exposes_verdict     = $14,
                    missing_referenced_media = $15,
                    check_completed          = $16,
                    media_origin             = $17,
                    is_rectified             = $18,
                    variant_id               = $19,
                    text_embedding           = $20,
                    gold_evidence_status     = $21,
                    gold_evidence_reason     = $22,
                    gold_evidence_updated_at = $23,
                    updated_at               = CURRENT_TIMESTAMP
                WHERE id = $24;
                """
        await self._execute(
            query,
            claim.data,
            claim.date,
            claim.language,
            claim.appearance_ids,
            claim.review_ids,
            claim.verdict_ids,
            claim.dismissed,
            claim.dismissed_reason,
            claim.rectifiable,
            claim.is_ambiguous,
            claim.is_inconsistent,
            claim.is_unshareable,
            claim.media_expose_verdict,
            claim.text_exposes_verdict,
            claim.missing_referenced_media,
            claim.check_completed,
            claim.media_origin,
            claim.is_rectified,
            claim.variant_id,
            claim.text_embedding,
            claim.gold_evidence_status,
            claim.gold_evidence_reason,
            claim.gold_evidence_updated_at,
            claim.id
        )

    async def insert_verdict(self, verdict: Verdict) -> int:
        """Adds the verdict to the database and returns the assigned ID.
        If there exists a verdict for the same claim, the existing verdict
        is kept but marked as is_current=False."""
        # Invalidate all previous verdicts for the same claim
        query = """UPDATE verdicts
                   SET is_current = FALSE
                   WHERE claim_id = $1"""
        await self._execute(query, verdict.claim_id)

        # Insert new verdict (clarity removed from schema)
        query = """
                INSERT INTO verdicts (claim_id, review_ids, media,
                                      veracity, veracity_explanation,
                                      context_coverage, context_coverage_explanation,
                                      integrity, full_verdict)
                VALUES ($1, $2, $3,
                        $4, $5,
                        $6, $7,
                        $8, $9)
                RETURNING id;
                """
        identifier = await self._fetchval(
            query,
            verdict.claim_id,
            verdict.review_ids,
            to_jsonb(verdict.media_verdicts),
            verdict.veracity.score if verdict.veracity else None,
            verdict.veracity.explanation if verdict.veracity else None,
            verdict.context_coverage.score if verdict.context_coverage else None,
            verdict.context_coverage.explanation if verdict.context_coverage else None,
            verdict.integrity.score,
            to_jsonb(verdict)
        )
        return identifier

    async def update_verdict(self, verdict: Verdict):
        """Updates an existing verdict in the database."""
        # raise RuntimeError("Verdicts should never be updated.")
        query = """
                UPDATE verdicts
                SET claim_id                  = $1,
                    review_ids                = $2,
                    media                     = $3,
                    veracity                  = $4,
                    veracity_explanation       = $5,
                    context_coverage          = $6,
                    context_coverage_explanation = $7,
                    integrity                 = $8,
                    full_verdict              = $9
                    WHERE id = $10;
        """
        await self._execute(
            query,
            verdict.claim_id,
            verdict.review_ids,
            to_jsonb(verdict.media_verdicts),
            verdict.veracity.score if verdict.veracity else None,
            verdict.veracity.explanation if verdict.veracity else None,
            verdict.context_coverage.score if verdict.context_coverage else None,
            verdict.context_coverage.explanation if verdict.context_coverage else None,
            verdict.integrity.score if verdict.integrity else None,
            to_jsonb(verdict),
            verdict.id
        )

    async def set_verdict_obsolete(self, verdict_id: int):
        """Mark a verdict as obsolete."""
        query = "UPDATE verdicts SET is_current = FALSE WHERE id = $1"
        await self._execute(query, verdict_id)

    async def update_integrity(self, verdict_id: int, integrity_score: float):
        """Updates the integrity score of a verdict."""
        query = "UPDATE verdicts SET integrity = $1 WHERE id = $2"
        await self._execute(query, integrity_score, verdict_id)

    # ------------------------------------------------------------------
    # Gold Evidence Reconstruction
    # ------------------------------------------------------------------

    async def insert_evidence(self, evidence: "Evidence") -> int:
        """Adds a reconstructed evidence item to the database and returns its ID.
        If an identical item (same claim, locator and proposition) already exists,
        that row is updated instead and its ID returned.

        Items without a locator match each other on claim and proposition alone,
        which is what `UNIQUE NULLS NOT DISTINCT` on `evidence` provides. On
        PostgreSQL below 15 they are not covered by the constraint and a repeated
        insert adds a second row; nothing in the pipeline does that, because
        extraction deduplicates in memory and re-extraction deletes first."""
        columns, values = _evidence_columns(evidence)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(values)))
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns
                            if c not in ("claim_id", "proposition", "proposition_hash",
                                         "source_locator", "source_locator_hash"))
        query = f"""
                INSERT INTO evidence ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT ON CONSTRAINT unique_evidence DO UPDATE
                    SET {updates}, updated_at = CURRENT_TIMESTAMP
                RETURNING id;
                """
        return await self._fetchval(query, *values)

    async def update_evidence(self, evidence: "Evidence") -> None:
        """Updates an existing evidence item."""
        assert evidence.id is not None, "Evidence must have an ID."
        columns, values = _evidence_columns(evidence)
        assignments = ", ".join(f"{c} = ${i + 1}" for i, c in enumerate(columns))
        query = (f"UPDATE evidence SET {assignments}, updated_at = CURRENT_TIMESTAMP "
                 f"WHERE id = ${len(values) + 1};")
        await self._execute(query, *values, evidence.id)

    async def get_evidence_by_id(self, evidence_id: int) -> "Evidence | None":
        row = await self._fetchrow("SELECT id, full_evidence FROM evidence WHERE id = $1", evidence_id)
        if row:
            return row_to_evidence(row)

    async def get_evidence_for_claim(self, claim_id: int,
                                     admissible_only: bool = False) -> list["Evidence"]:
        """Returns all reconstructed evidence items of a claim, oldest ID first."""
        query = "SELECT id, full_evidence FROM evidence WHERE claim_id = $1"
        if admissible_only:
            query += " AND admissible IS TRUE"
        query += " ORDER BY id;"
        rows = await self._fetch(query, claim_id)
        return [row_to_evidence(row) for row in rows]

    async def get_evidence_for_claims(self, claim_ids: list[int]) -> dict[int, list["Evidence"]]:
        """Bulk variant of `get_evidence_for_claim` for the analysis scripts."""
        if not claim_ids:
            return {}
        rows = await self._fetch(
            "SELECT id, claim_id, full_evidence FROM evidence WHERE claim_id = ANY($1) ORDER BY id;",
            claim_ids,
        )
        result: dict[int, list] = {claim_id: [] for claim_id in claim_ids}
        for row in rows:
            result[row["claim_id"]].append(row_to_evidence(row))
        return result

    async def delete_evidence_for_claim(self, claim_id: int) -> int:
        """Removes all reconstructed evidence of a claim and returns how many rows
        were deleted. Used before a re-extraction, so that items the new run no
        longer produces do not survive as orphans."""
        result = await self._execute("DELETE FROM evidence WHERE claim_id = $1;", claim_id)
        try:
            return int(str(result).split()[-1])
        except (ValueError, IndexError, TypeError):
            return 0

    async def count_evidence_for_claim(self, claim_id: int) -> int:
        return await self._fetchval("SELECT COUNT(*) FROM evidence WHERE claim_id = $1", claim_id)

    async def set_gold_evidence_status(self, claim_id: int, status: str,
                                       reason: str | None = None) -> None:
        """Records the Gold Evidence Reconstruction outcome on the claim.

        Touches only the three additive `gold_evidence_*` columns. In particular,
        `dismissed` is deliberately left untouched: a rejected instance stays a
        valid VeriTaS claim."""
        query = """
                UPDATE claims
                SET gold_evidence_status     = $1,
                    gold_evidence_reason     = $2,
                    gold_evidence_updated_at = CURRENT_TIMESTAMP
                WHERE id = $3;
                """
        await self._execute(query, status, reason, claim_id)

    async def save_gold_evidence_result(self, result: dict) -> int:
        """Upserts the outcome of the sufficiency validator for one
        (claim, condition, ensemble_mode) triple."""
        query = """
                INSERT INTO gold_evidence_results (claim_id, condition, ensemble_mode, n_evidence,
                                                   predicted_verdict, member_responses, property_diffs,
                                                   max_property_diff, is_close, threshold,
                                                   model_specifiers, error)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT ON CONSTRAINT unique_gold_evidence_result DO UPDATE
                    SET n_evidence        = EXCLUDED.n_evidence,
                        predicted_verdict = EXCLUDED.predicted_verdict,
                        member_responses  = EXCLUDED.member_responses,
                        property_diffs    = EXCLUDED.property_diffs,
                        max_property_diff = EXCLUDED.max_property_diff,
                        is_close          = EXCLUDED.is_close,
                        threshold         = EXCLUDED.threshold,
                        model_specifiers  = EXCLUDED.model_specifiers,
                        error             = EXCLUDED.error,
                        updated_at        = CURRENT_TIMESTAMP
                RETURNING id;
                """
        return await self._fetchval(
            query,
            result["claim_id"],
            result["condition"],
            result["ensemble_mode"],
            result.get("n_evidence", 0),
            to_jsonb(result.get("predicted_verdict")),
            to_jsonb(result.get("member_responses")),
            to_jsonb(result.get("property_diffs")),
            result.get("max_property_diff"),
            result.get("is_close"),
            result.get("threshold"),
            result.get("model_specifiers"),
            result.get("error"),
        )

    async def get_gold_evidence_results(self, claim_id: int,
                                        ensemble_mode: str | None = None) -> list[dict]:
        query = "SELECT * FROM gold_evidence_results WHERE claim_id = $1"
        args = [claim_id]
        if ensemble_mode:
            query += " AND ensemble_mode = $2"
            args.append(ensemble_mode)
        rows = await self._fetch(query + ";", *args)
        return [dict(row) for row in rows]

    async def get_all_gold_evidence_results(self, ensemble_mode: str | None = None) -> list[dict]:
        if ensemble_mode:
            rows = await self._fetch(
                "SELECT * FROM gold_evidence_results WHERE ensemble_mode = $1 ORDER BY claim_id;",
                ensemble_mode)
        else:
            rows = await self._fetch("SELECT * FROM gold_evidence_results ORDER BY claim_id;")
        return [dict(row) for row in rows]

    async def get_claims_for_gold_evidence(
            self,
            limit: int | None = None,
            start_date: date | datetime | None = None,
            end_date: date | datetime | None = None,
            statuses: list[str] | None = None,
            released_first: bool = True,
            claim_ids: list[int] | None = None,
            exclude_claim_ids: list[int] | None = None,
    ) -> list[Claim]:
        """Returns candidate claims for the Gold Evidence Reconstruction.

        A candidate is a non-dismissed claim with a current verdict whose reviews
        completed at least stage 6 (7 for rectified claims), i.e. the gold verdict
        is final. Released claims are processed first; if none are left, the
        remaining verdict-complete claims follow.

        `statuses` filters on `gold_evidence_status`; pass `[None]`-like values via
        the string 'unprocessed' to select claims that were never processed.
        `exclude_claim_ids` skips claims a caller has already handled - a run that
        batches through the resumable statuses needs it, because a claim it just
        deferred still matches the filter.
        """
        if isinstance(start_date, date) and not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())
        if isinstance(end_date, date) and not isinstance(end_date, datetime):
            end_date = datetime.combine(end_date, datetime.max.time())

        wants_unprocessed = bool(statuses) and "unprocessed" in statuses
        concrete_statuses = [s for s in (statuses or []) if s != "unprocessed"] or None

        # Both flags are ordered separately rather than as `a OR b`: with
        # SELECT DISTINCT, PostgreSQL only accepts ORDER BY expressions that appear
        # in the select list, and this yields the same released-first grouping.
        order = ("c.released_quarter DESC, c.released_longitudinal DESC, c.id"
                 if released_first else "c.id")

        query = f"""
                SELECT DISTINCT c.*
                FROM claims c
                         JOIN verdicts v ON v.claim_id = c.id AND v.is_current
                         JOIN reviews r ON r.id = ANY (c.review_ids)
                WHERE c.dismissed = FALSE
                  AND r.dismissed = FALSE
                  AND r.stage >= CASE WHEN c.is_rectified THEN 7 ELSE 6 END
                  AND ($2::timestamp IS NULL OR c.date >= $2)
                  AND ($3::timestamp IS NULL OR c.date <= $3)
                  AND ($4::text[] IS NULL OR c.gold_evidence_status = ANY ($4)
                       OR ($5::bool AND c.gold_evidence_status IS NULL))
                  AND ($6::int[] IS NULL OR c.id = ANY ($6))
                  AND ($7::int[] IS NULL OR NOT (c.id = ANY ($7)))
                ORDER BY {order}
                LIMIT $1;
                """
        rows = await self._fetch(query, limit, start_date, end_date,
                                 concrete_statuses, wants_unprocessed, claim_ids,
                                 exclude_claim_ids or None)
        return [Claim.model_validate(dict(row)) for row in rows]

    async def insert(self, instance: VeritasBaseModel) -> int:
        """Inserts any (compatible) object into the database and returns the assigned ID."""
        from veritas.gold_evidence.models import Evidence

        match instance:
            case Evidence():
                return await self.insert_evidence(instance)
            case Review():
                return await self.insert_review(instance)
            case Publisher():
                return await self.insert_publisher(instance)
            case Article():
                return await self.insert_article(instance)
            case Appearance():
                return await self.insert_appearance(instance)
            case Claim():
                return await self.insert_claim(instance)
            case Verdict():
                return await self.insert_verdict(instance)
            case _:
                raise ValueError(f"Unsupported object type: {type(instance)}")

    async def update(self, instance: VeritasBaseModel):
        """Updates any (compatible) object into the database."""
        from veritas.gold_evidence.models import Evidence

        assert instance.id is not None, "Object must have an ID."
        match instance:
            case Evidence():
                return await self.update_evidence(instance)
            case Review():
                return await self.update_review(instance)
            case Publisher():
                return await self.update_publisher(instance)
            case Article():
                return await self.update_article(instance)
            case Appearance():
                return await self.update_appearance(instance)
            case Claim():
                return await self.update_claim(instance)
            case Verdict():
                return await self.update_verdict(instance)
            case _:
                raise ValueError(f"Unsupported object type: {type(instance)}")

    async def get(self, cls, id: int) -> VeritasBaseModel | None:
        from veritas.gold_evidence.models import Evidence

        if cls is Evidence:
            return await self.get_evidence_by_id(id)
        if cls is Review:
            return await self.get_review_by_id(id)
        elif cls is Publisher:
            return await self.get_publisher_by_id(id)
        elif cls is Article:
            return await self.get_article_by_id(id)
        elif cls is Appearance:
            return await self.get_appearance_by_id(id)
        elif cls is Claim:
            return await self.get_claim_by_id(id)
        elif cls is Verdict:
            return await self.get_verdict_by_id(id)
        else:
            raise ValueError(f"Unsupported object type: {cls}")

    async def get_claim_reviews(self, url_claim_pairs: list[tuple[str, str]], source: str) -> list[dict]:
        """Returns the raw ClaimReviews for the reviews specified by the given
        URL-claim pairs. Uses an SQL trick to make it faster than individual
        queries. `source` is either 'google' or 'datacommons'."""
        # Construct a temporary table to handle the mass of URL-claim pairs
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    CREATE TEMP TABLE temp_input_claims
                    (
                        id    SERIAL,
                        url   TEXT,
                        claim TEXT
                    ) ON COMMIT DROP;
                    """
                )

                # Bulk insert data into temp table using COPY
                await conn.copy_records_to_table(
                    "temp_input_claims", records=url_claim_pairs, columns=["url", "claim"]
                )

                # Join and fetch matching ClaimReviews
                rows = await conn.fetch(
                    f"""
                    SELECT r.{source}_claim_review
                    FROM reviews r
                    RIGHT JOIN temp_input_claims t
                    ON r.url = t.url AND r.raw_claim = t.claim
                    ORDER BY t.id;
                """
                )

        return list(row[f"{source}_claim_review"] for row in rows)

    async def get_review_by_id(self, id: int) -> Review | None:
        """Retrieve a review by ID."""
        query = "SELECT * FROM reviews WHERE id = $1"
        row = await self._fetchrow(query, id)
        if row:
            return Review.model_validate(dict(row))

    async def get_review_by_url(self, url: HttpUrl) -> Review | None:
        """Retrieve a review by URL. The URL must match exactly except
        for trailing slashes."""
        if not url:
            return None
        query = "SELECT * FROM reviews WHERE RTRIM(url, '/') = RTRIM($1, '/')"
        row = await self._fetchrow(query, str(url))
        if row:
            return Review.model_validate(dict(row))

    async def get_reviews_for_appearance(self, appearance_id: int) -> list[Review]:
        """Retrieves all reviews featuring the specified appearance."""
        query = "SELECT * FROM reviews WHERE $1 = ANY(appearance_ids)"
        rows = await self._fetch(query, appearance_id)
        return [Review.model_validate(dict(row)) for row in rows if row]

    async def get_all_reviews(self) -> list[Review]:
        """Retrieves all reviews."""
        query = "SELECT * FROM reviews"
        rows = await self._fetch(query)
        return [Review.model_validate(dict(row)) for row in rows if row]

    async def get_reviews(
            self,
            stage: int,
            limit: int | None = None,
            language: str | None = None,
            dismissed: bool = False,
            start_date: date | datetime | None = None,
            end_date: date | datetime | None = None,
    ) -> list[Review]:
        """Returns all reviews as specified by the given parameters."""
        query = """
                SELECT r.*
                FROM reviews r
                WHERE r.stage = $1
                  AND (r.deferred_until IS NULL OR r.deferred_until <= CURRENT_TIMESTAMP)
                  AND ( -- Filter for language
                    $2::text IS NULL OR r.language = $2
                    )
                  AND ( -- Filter dismissed
                    $4::bool IS NULL OR r.dismissed = $4::bool
                    )
                  AND ( -- Filter by published start date (inclusive)
                    $5::timestamp IS NULL OR r.published >= $5
                    )
                  AND ( -- Filter by published end date (inclusive)
                    $6::timestamp IS NULL OR r.published <= $6
                    )
                LIMIT $3;
                """
        # Normalize possible date inputs to timestamps
        if isinstance(start_date, date) and not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())
        if isinstance(end_date, date) and not isinstance(end_date, datetime):
            end_date = datetime.combine(end_date, datetime.max.time())

        rows = await self._fetch(query, stage, language, limit, dismissed, start_date, end_date)

        return [Review.model_validate(dict(row)) for row in rows]

    async def get_reviews_by_publisher_id(self, publisher_id: int) -> list[Review]:
        """Retrieves all reviews for a specific publisher."""
        query = "SELECT * FROM reviews WHERE publisher_id = $1"
        rows = await self._fetch(query, publisher_id)
        return [Review.model_validate(dict(row)) for row in rows if row]

    async def count_claims(
            self,
            start_date: date | datetime | None = None,
            end_date: date | datetime | None = None,
    ) -> int:
        """Counts the total number of claims (incl. dismissed) for a given stage and date range."""
        query = """
                SELECT COUNT(*)
                FROM claims
                WHERE ($1::timestamp IS NULL OR date >= $1)
                  AND ($2::timestamp IS NULL OR date <= $2);
                """
        # Normalize possible date inputs to timestamps
        if isinstance(start_date, date) and not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())
        if isinstance(end_date, date) and not isinstance(end_date, datetime):
            end_date = datetime.combine(end_date, datetime.max.time())

        return await self._fetchval(query, start_date, end_date)

    async def get_publisher_by_id(self, publisher_id: int) -> Publisher | None:
        """Retrieve a publisher by ID."""
        query = "SELECT * FROM publishers WHERE id = $1"
        row = await self._fetchrow(query, publisher_id)
        if row:
            return Publisher.model_validate(dict(row))

    async def get_publisher_by_url(self, url: str) -> Publisher | None:
        """Retrieve a publisher by URL. The URL may be a substring (e.g.,
        just the domain) of the publisher's actual URL."""
        if not url:
            return None
        domain = get_domain(url)
        query = "SELECT * FROM publishers WHERE $1 = ANY(domains);"
        row = await self._fetchrow(query, domain)
        if row:
            return Publisher.model_validate(dict(row))

    async def get_publisher_by_name(self, name: str) -> Publisher | None:
        """Retrieve a publisher by exact match name."""
        query = "SELECT * FROM publishers WHERE name = $1;"
        row = await self._fetchrow(query, name)
        if row:
            return Publisher.model_validate(dict(row))

    async def get_article_by_id(self, article_id: int) -> Article | None:
        """Retrieve an article by ID."""
        query = "SELECT * FROM articles WHERE id = $1"
        row = await self._fetchrow(query, article_id)
        if row:
            return Article.model_validate(dict(row))

    async def get_article_by_url(self, url: HttpUrl) -> Article | None:
        """Retrieves the article specified by the given URL. The URL must
        match exactly except for trailing slashes."""
        if not url:
            return None
        query = "SELECT * FROM articles WHERE RTRIM(url, '/') = RTRIM($1, '/')"
        row = await self._fetchrow(query, str(url))
        if row:
            return Article.model_validate(dict(row))

    async def article_exists(self, url: HttpUrl) -> bool:
        """Checks if an article with the given URL exists in the database. The URL must
        match exactly except for trailing slashes."""
        if not url:
            return False
        query = "SELECT EXISTS(SELECT 1 FROM articles WHERE RTRIM(url, '/') = RTRIM($1, '/'))"
        return await self._fetchval(query, str(url))

    async def get_appearance_by_id(self, appearance_id: int) -> Appearance | None:
        """Retrieve a claim appearance by ID."""
        query = "SELECT * FROM appearances WHERE id = $1"
        row = await self._fetchrow(query, appearance_id)
        if row:
            row = dict(row)
            # Convert content columns
            try:
                if "original_scraped_content" in row:
                    row["original_scraped_content"] = (
                        MultimodalSequence(row["original_scraped_content"]) if row["original_scraped_content"] else None
                    )
                if "archived_scraped_content" in row:
                    row["archived_scraped_content"] = (
                        MultimodalSequence(row["archived_scraped_content"]) if row["archived_scraped_content"] else None
                    )
            except ValueError as e:
                logger.warning(f"Broken reference in appearance {appearance_id} scraped_content columns: {e}")
                row["original_scraped_content"] = None
                row["archived_scraped_content"] = None
            return Appearance.model_validate(row)

    async def get_appearance_by_url(self, url: HttpUrl) -> Appearance | None:
        """Retrieves the appearance specified by the given URL."""
        query = "SELECT * FROM appearances WHERE url = $1"
        row = await self._fetchrow(query, str(url))
        if row:
            row = dict(row)
            try:
                if "original_scraped_content" in row:
                    row["original_scraped_content"] = (
                        MultimodalSequence(row["original_scraped_content"]) if row["original_scraped_content"] else None
                    )
                if "archived_scraped_content" in row:
                    row["archived_scraped_content"] = (
                        MultimodalSequence(row["archived_scraped_content"]) if row["archived_scraped_content"] else None
                    )
            except ValueError as e:
                logger.warning(f"Broken reference in appearance (url={url}) scraped_content columns: {e}")
                row["original_scraped_content"] = None
                row["archived_scraped_content"] = None
            return Appearance.model_validate(row)

    async def get_appearance_by_archive_url(self, archive_url: HttpUrl) -> Appearance | None:
        """Retrieves the appearance specified by the given archive URL."""
        query = "SELECT * FROM appearances WHERE archive_url = $1;"
        row = await self._fetchrow(query, str(archive_url))
        if row:
            row = dict(row)
            try:
                if "original_scraped_content" in row:
                    row["original_scraped_content"] = (
                        MultimodalSequence(row["original_scraped_content"]) if row["original_scraped_content"] else None
                    )
                if "archived_scraped_content" in row:
                    row["archived_scraped_content"] = (
                        MultimodalSequence(row["archived_scraped_content"]) if row["archived_scraped_content"] else None
                    )
            except ValueError as e:
                logger.warning(
                    f"Broken reference in appearance (archive_url={archive_url}) scraped_content columns: {e}")
                row["original_scraped_content"] = None
                row["archived_scraped_content"] = None
            return Appearance.model_validate(row)

    async def replace_appearance(self, to_replace_id: int, replacement_id: int):
        """Replaces the appearance (ID to_replace_id) with the appearance (ID replacement_id)."""
        if to_replace_id == replacement_id:
            return

        # 1) Reviews: if replacement not present -> replace directly
        await self._execute(
            """
            UPDATE reviews
            SET appearance_ids = array_replace(appearance_ids, $1, $2)
            WHERE appearance_ids @> ARRAY [$1]::integer[]
              AND NOT (appearance_ids @> ARRAY [$2]::integer[]);
            """,
            to_replace_id,
            replacement_id,
        )

        # 2) Reviews: if both present -> remove the duplicate old id
        await self._execute(
            """
            UPDATE reviews
            SET appearance_ids = array_remove(appearance_ids, $1)
            WHERE appearance_ids @> ARRAY [$1]::integer[]
              AND (appearance_ids @> ARRAY [$2]::integer[]);
            """,
            to_replace_id,
            replacement_id,
        )

        # 3) Claims: if replacement not present -> replace directly
        await self._execute(
            """
            UPDATE claims
            SET appearance_ids = array_replace(appearance_ids, $1, $2)
            WHERE appearance_ids @> ARRAY [$1]::integer[]
              AND NOT (appearance_ids @> ARRAY [$2]::integer[]);
            """,
            to_replace_id,
            replacement_id,
        )

        # 4) Claims: if both present -> remove the duplicate old id
        await self._execute(
            """
            UPDATE claims
            SET appearance_ids = array_remove(appearance_ids, $1)
            WHERE appearance_ids @> ARRAY [$1]::integer[]
              AND (appearance_ids @> ARRAY [$2]::integer[]);
            """,
            to_replace_id,
            replacement_id,
        )

    async def get_claim_by_id(self, claim_id: int) -> Claim | None:
        """Retrieve a claim by ID."""
        query = "SELECT * FROM claims WHERE id = $1"
        row = await self._fetchrow(query, claim_id)
        if row:
            return Claim.model_validate(dict(row))

    async def get_claims_by_ids(self, claim_ids: list[int]) -> list[Claim]:
        """Retrieve multiple claims by IDs in a single query."""
        if not claim_ids:
            return []
        query = "SELECT * FROM claims WHERE id = ANY($1) ORDER BY id"
        rows = await self._fetch(query, claim_ids)
        return [Claim.model_validate(dict(row)) for row in rows]

    async def get_n_claims(self) -> int:
        """Returns the number of claims in the database."""
        query = "SELECT COUNT(*) FROM claims"
        return await self._fetchval(query)

    async def get_claim_id_by_data(self, data: MultimodalSequence) -> int | None:
        """Retrieve a claim's ID by its data."""
        query = "SELECT id FROM claims WHERE data = $1"
        row = await self._fetchrow(query, str(data))
        if row:
            return row[0]

    async def get_claims_without_verdicts(
            self,
            limit: int | None = None,
            start_date: date | datetime | None = None,
            end_date: date | datetime | None = None,
            is_rectified: bool | None = None
    ) -> list[Claim]:
        """Retrieve the list of non-dismissed claims that don't have a verdict yet.
        Optionally restrict by claim date within [start_date, end_date] (inclusive)."""
        # Normalize dates to timestamps for Postgres comparison
        if isinstance(start_date, date) and not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())
        if isinstance(end_date, date) and not isinstance(end_date, datetime):
            end_date = datetime.combine(end_date, datetime.max.time())

        query = """
                SELECT *
                FROM claims
                WHERE verdict_ids = '{}'
                  AND dismissed = FALSE
                  AND check_completed = TRUE
                  AND ($2::timestamp IS NULL OR date >= $2)
                  AND ($3::timestamp IS NULL OR date <= $3)
                  AND ($4::bool IS NULL OR is_rectified = $4)
                ORDER BY id
                LIMIT $1;
                """
        rows = await self._fetch(query, limit, start_date, end_date, is_rectified)
        return [Claim.model_validate(dict(row)) for row in rows]

    async def get_claims_to_rectify(
            self,
            start_date: date,
            end_date: date,
            limit: int | None = None,
    ) -> list[Claim]:
        """Retrieve a list of claims from the specified time period that need to be rectified.
        These are the claims which:
        1. Are original.
        2. Weren't rectified yet.
        3. Have corrupted integrity (with rather high certainty).
        4. Weren't dismissed yet.
        5. Are rectifiable.
        end_date is inclusive."""
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
        query = """
                SELECT c.*
                FROM claims c
                         LEFT JOIN verdicts v ON c.id = v.claim_id
                WHERE c.date BETWEEN $1 AND $2
                  AND c.is_rectified = FALSE -- Condition 1
                  AND c.variant_id IS NULL   -- Condition 2
                  AND v.integrity < -1 / 3   -- Condition 3
                  AND v.is_current = TRUE    -- Condition 3
                  AND c.dismissed = FALSE    -- Condition 4
                  AND (c.rectifiable IS NULL OR c.rectifiable != FALSE) -- Condition 5
                LIMIT $3;
                """
        rows = await self._fetch(query, start_date, end_date, limit)
        return [Claim.model_validate(dict(row)) for row in rows]

    async def get_verdicts(self, only_released: bool = False) -> list[Verdict]:
        """Retrieves all verdicts in the database.

        If *only_released* is True, only verdicts whose claim has been released
        (released_quarter OR released_longitudinal) are returned.
        """
        if only_released:
            query = """
                    SELECT v.*
                    FROM verdicts v
                    JOIN claims c ON v.claim_id = c.id
                    WHERE v.is_current
                      AND (c.released_quarter OR c.released_longitudinal)
                    ORDER BY v.id;
                    """
        else:
            query = """
                    SELECT *
                    FROM verdicts
                    WHERE is_current
                    ORDER BY id;
                    """
        rows = await self._fetch(query)
        return [row_to_verdict(row) for row in rows]

    async def get_claims_by_language(self, language: str, released: bool = None) -> list[Claim]:
        """Retrieves all non-dismissed claims for a given language."""
        query = "SELECT * FROM claims WHERE language = $1 AND dismissed = FALSE"
        if released is not None:
            query += " AND released_quarter = $2"
        rows = await self._fetch(query, language, released) if released is not None else await self._fetch(query, language)
        return [Claim.model_validate(dict(row)) for row in rows]

    async def get_verdict_by_id(self, verdict_id: int) -> Verdict | None:
        """Retrieve a verdict by ID."""
        query = "SELECT * FROM verdicts WHERE id = $1"
        row = await self._fetchrow(query, verdict_id)
        if row:
            return Verdict.model_validate(dict(row["full_verdict"]))

    async def get_verdict_by_claim_id(self, claim_id: int) -> Verdict | None:
        """Retrieve the current verdict of a claim specified by its ID."""
        query = "SELECT id, full_verdict FROM verdicts WHERE claim_id = $1 AND is_current = TRUE"
        row = await self._fetchrow(query, claim_id)
        if row:
            return row_to_verdict(row)

    async def count_intact_claims(self, start_date: date = None, end_date: date = None) -> int:
        """Counts the number of intact claims within the specified date range."""
        query = """
                SELECT COUNT(*) as claim_count
                FROM claims c
                         LEFT JOIN verdicts v ON c.id = v.claim_id
                WHERE c.dismissed = FALSE
                  AND v.integrity > 1 / 3
                  AND ($1::date IS NULL OR c.date >= $1)
                  AND ($2::date IS NULL OR c.date <= $2);
                """
        rows = await self._fetch(query, start_date, end_date)
        return rows[0]["claim_count"]

    async def count_claims_with_verdicts(
            self,
            start_date: date = None,
            end_date: date = None,
    ):
        """Counts the number of claims that have a verdict within the specified date range."""
        query = """
                SELECT COUNT(*) as claim_count
                FROM claims
                WHERE verdict_ids != '{}'
                  AND dismissed = FALSE
                  AND ($1::date IS NULL OR date >= $1)
                  AND ($2::date IS NULL OR date <= $2);
                """
        rows = await self._fetch(query, start_date, end_date)
        return rows[0]["claim_count"]

    async def get_stage_summary(self) -> dict[int, int]:
        """Computes the number of non-dismissed reviews per stage."""
        query = """
                SELECT stage, COUNT(*) as review_count
                FROM reviews
                WHERE dismissed = FALSE
                GROUP BY stage;
                """
        rows = await self._fetch(query)
        return {row["stage"]: row["review_count"] for row in rows}

    async def count_dismissed_reviews(self) -> int:
        """Returns the number of dismissed reviews in the database."""
        query = "SELECT COUNT(*) FROM reviews WHERE dismissed = TRUE"
        return await self._fetchval(query)

    async def get_dismissed_reasons(self) -> Counter:
        """Returns all unique dismissed reasons and their counts."""
        query = """
                SELECT dismissed_reason, COUNT(*) as count
                FROM reviews
                WHERE dismissed = TRUE
                GROUP BY dismissed_reason;
                """
        rows = await self._fetch(query)
        counts = {str(row["dismissed_reason"]): int(row["count"]) for row in rows}
        counts = Counter(counts)
        return counts

    async def get_media_embedding(self, kind: str, media_id: int) -> list[float] | None:
        """Returns the embedding of a media item."""
        query = """
                SELECT embedding FROM media
                WHERE kind = $1 AND media_id = $2;
                """
        return await self._fetchval(query, kind, media_id)

    async def get_all_embeddings(self, kind: str) -> list[tuple[int, list[float]]]:
        """Returns all embeddings of a kind (text, image, video)."""
        if kind == "text":
            query = """SELECT id, text_embedding AS embedding
                       FROM claims
                       WHERE text_embedding IS NOT NULL
                           AND (released_longitudinal OR released_quarter)"""
            rows = await self._fetch(query)
        else:
            # `media` also holds media that were only indexed, never embedded.
            query = """SELECT media_id AS id, embedding
                       FROM media
                       WHERE kind = $1
                         AND embedding IS NOT NULL"""
            rows = await self._fetch(query, kind)
        return [(row["id"], row["embedding"]) for row in rows]

    # -- Media index --------------------------------------------------------

    async def rebuild_media_index(self) -> int:
        """Rebuilds the usage columns of `media` from every place a medium can
        occur in VeriTaS, and returns the number of media that are in use.

        Scanned are `claims.data`, both scraped contents of an appearance, an
        article's raw page and extracted content, and the gold evidence's raw
        source content.

        The scan runs entirely inside PostgreSQL, so none of the (potentially
        huge) scraped content ever has to cross the wire - but it does read every
        such column once, which takes a while on a grown database. Reset and
        re-insert share one transaction: an interrupted run must never leave the
        index empty, because an empty index looks exactly like "nothing is in use".
        """
        reset_query = """
                      UPDATE media
                      SET claim_ids      = '{}',
                          appearance_ids = '{}',
                          article_ids    = '{}',
                          evidence_ids   = '{}',
                          indexed_at     = CURRENT_TIMESTAMP;
                      """

        # `regexp_matches` is strict, so a NULL column simply contributes no rows.
        index_query = f"""
            WITH refs AS (
                SELECT 'claim'::TEXT AS source, c.id AS source_id, ref[1] AS kind, ref[2]::INTEGER AS media_id
                FROM claims c, LATERAL regexp_matches(c.data, $1, 'g') AS m(ref)
                UNION ALL
                SELECT 'appearance', a.id, ref[1], ref[2]::INTEGER
                FROM appearances a, LATERAL regexp_matches(a.original_scraped_content, $1, 'g') AS m(ref)
                UNION ALL
                SELECT 'appearance', a.id, ref[1], ref[2]::INTEGER
                FROM appearances a, LATERAL regexp_matches(a.archived_scraped_content, $1, 'g') AS m(ref)
                UNION ALL
                SELECT 'article', ar.id, ref[1], ref[2]::INTEGER
                FROM articles ar, LATERAL regexp_matches(ar.scraped_page, $1, 'g') AS m(ref)
                UNION ALL
                SELECT 'article', ar.id, ref[1], ref[2]::INTEGER
                FROM articles ar, LATERAL regexp_matches(ar.extracted_article, $1, 'g') AS m(ref)
                UNION ALL
                SELECT 'evidence', e.id, ref[1], ref[2]::INTEGER
                FROM evidence e, LATERAL regexp_matches(e.source_raw_content, $1, 'g') AS m(ref)
            ),
                 aggregated AS (
                     SELECT kind,
                            media_id,
                            {_usage_agg("claim")}      AS claim_ids,
                            {_usage_agg("appearance")} AS appearance_ids,
                            {_usage_agg("article")}    AS article_ids,
                            {_usage_agg("evidence")}   AS evidence_ids
                     FROM refs
                     GROUP BY kind, media_id
                 )
            INSERT INTO media (kind, media_id, claim_ids, appearance_ids, article_ids,
                               evidence_ids, indexed_at)
            SELECT kind, media_id, claim_ids, appearance_ids, article_ids, evidence_ids,
                   CURRENT_TIMESTAMP
            FROM aggregated
            ON CONFLICT (kind, media_id) DO UPDATE
            SET claim_ids      = EXCLUDED.claim_ids,
                appearance_ids = EXCLUDED.appearance_ids,
                article_ids    = EXCLUDED.article_ids,
                evidence_ids   = EXCLUDED.evidence_ids,
                indexed_at     = EXCLUDED.indexed_at;
            """

        async with self._transaction() as conn:
            await conn.execute(reset_query)
            await conn.execute(index_query, MEDIA_REF_PATTERN)
            return await conn.fetchval(f"SELECT COUNT(*) FROM media WHERE {MEDIA_IS_USED};")

    async def get_media_index(self) -> dict[str, set[int]]:
        """The IDs of all media that occur somewhere in VeriTaS, keyed by kind.

        Media that are merely known to the table (e.g. because an embedding was
        computed for them once) but occur nowhere are deliberately left out: this
        is exactly the set of files the media cleanup keeps on disk.
        """
        query = f"SELECT kind, media_id FROM media WHERE {MEDIA_IS_USED};"
        index: dict[str, set[int]] = {}
        for row in await self._fetch(query):
            index.setdefault(row["kind"], set()).add(row["media_id"])
        return index

    async def get_media_usage(self, kind: str, media_id: int) -> dict | None:
        """Where a single medium is used, or None if it is unknown to the index."""
        query = """
                SELECT kind, media_id, claim_ids, appearance_ids, article_ids, evidence_ids,
                       embedding IS NOT NULL AS has_embedding, indexed_at
                FROM media
                WHERE kind = $1 AND media_id = $2;
                """
        row = await self._fetchrow(query, kind, media_id)
        return dict(row) if row else None

    async def get_media_index_stats(self) -> dict:
        """Counts describing the media index: media per kind, and how many
        claims, appearances, articles and evidence items refer to media."""
        per_kind_query = f"""
            SELECT kind,
                   COUNT(*)                                                AS n_media,
                   COUNT(*) FILTER (WHERE {MEDIA_IS_USED})                 AS n_used,
                   COUNT(*) FILTER (WHERE CARDINALITY(claim_ids) > 0)      AS n_in_claims,
                   COUNT(*) FILTER (WHERE CARDINALITY(appearance_ids) > 0) AS n_in_appearances,
                   COUNT(*) FILTER (WHERE CARDINALITY(article_ids) > 0)    AS n_in_articles,
                   COUNT(*) FILTER (WHERE CARDINALITY(evidence_ids) > 0)   AS n_in_evidence,
                   COUNT(*) FILTER (WHERE embedding IS NOT NULL)           AS n_with_embedding
            FROM media
            GROUP BY kind
            ORDER BY kind;
            """
        sources_query = """
            SELECT (SELECT COUNT(*) FROM (SELECT DISTINCT UNNEST(claim_ids) FROM media) t)      AS n_claims,
                   (SELECT COUNT(*) FROM (SELECT DISTINCT UNNEST(appearance_ids) FROM media) t) AS n_appearances,
                   (SELECT COUNT(*) FROM (SELECT DISTINCT UNNEST(article_ids) FROM media) t)    AS n_articles,
                   (SELECT COUNT(*) FROM (SELECT DISTINCT UNNEST(evidence_ids) FROM media) t)   AS n_evidence;
            """
        per_kind = [dict(row) for row in await self._fetch(per_kind_query)]
        sources = dict(await self._fetchrow(sources_query))
        return {
            "per_kind": per_kind,
            "n_media": sum(row["n_media"] for row in per_kind),
            "n_used": sum(row["n_used"] for row in per_kind),
            "n_with_embedding": sum(row["n_with_embedding"] for row in per_kind),
            "sources": sources,
        }


database, user, password, host, port = database.values()
db = VeritasDB(database=database, user=user, password=password, host=host, port=port)


def to_jsonb(obj):
    """Format when saved into the database."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    elif isinstance(obj, list):
        return [to_jsonb(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_jsonb(value) for key, value in obj.items()}
    else:
        return obj


def row_to_verdict(row) -> Verdict:
    """Format when retrieved from the database."""
    row = dict(row)
    verdict = Verdict.model_validate(row["full_verdict"])
    if "id" in row:
        verdict.id = row["id"]
    return verdict


def row_to_evidence(row):
    """Reconstructs an Evidence object from its JSONB round-trip blob."""
    from veritas.gold_evidence.models import Evidence

    row = dict(row)
    evidence = Evidence.model_validate(row["full_evidence"])
    if row.get("id") is not None:
        evidence.id = row["id"]
    return evidence


def _evidence_columns(evidence) -> tuple[list[str], list]:
    """Flattens an Evidence object into (column names, values) for SQL.

    The flat columns exist for querying/aggregation; `full_evidence` is the
    authoritative round-trip representation (same pattern as `verdicts`)."""
    faithfulness = evidence.faithfulness
    temporal = evidence.temporal_validation
    data = {
        "claim_id": evidence.claim_id,
        "review_id": evidence.review_id,
        "article_id": evidence.article_id,
        "proposition": evidence.proposition,
        "proposition_hash": hash_int32(evidence.proposition),
        "source_name": evidence.source.name,
        "source_kind": evidence.source.kind.value,
        # Both stay NULL for sources that are not publications (a tool without a
        # page, offline evidence); see the uniqueness constraint on `evidence`.
        "source_locator": evidence.source.locator,
        "source_locator_hash": (hash_int32(evidence.source.locator)
                                if evidence.source.locator else None),
        "source_proximity": evidence.source.proximity.value,
        "source_raw_content": evidence.source.raw_content,
        "available_since": evidence.available_since,
        "role": evidence.role.value,
        "accessed_at": evidence.accessed_at,
        "extraction_reasoning": evidence.extraction_reasoning,
        "extraction_confidence": evidence.extraction_confidence,
        "accessible": evidence.accessible,
        "faithfulness_assessment": faithfulness.assessment if faithfulness else None,
        # `reasoning` is the provider's reasoning trace, `justification` the
        # short reason the model was asked to state.
        "faithfulness_reasoning": faithfulness.reasoning if faithfulness else None,
        "faithfulness_justification": faithfulness.justification if faithfulness else None,
        "before_fact_check": temporal.before_fact_check if temporal else None,
        "before_claim": temporal.before_claim if temporal else None,
        "later_event": temporal.later_event if temporal else None,
        "temporal_reasoning": temporal.reasoning if temporal else None,
        "temporal_justification": temporal.justification if temporal else None,
        "admissible": evidence.admissible,
        "inadmissibility_reason": evidence.inadmissibility_reason,
        "dismissed": evidence.dismissed,
        "dismissed_reason": evidence.dismissed_reason,
        "deferred_until": evidence.deferred_until,
        "full_evidence": to_jsonb(evidence),
    }
    return list(data.keys()), list(data.values())
