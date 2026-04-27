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

        # Media embeddings
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS media_embeddings
            (
                id         SERIAL PRIMARY KEY,
                kind       TEXT    NOT NULL,
                media_id   INTEGER NOT NULL,
                embedding  FLOAT[] NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (kind, media_id)
            );
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
                                    is_rectified, variant_id, text_embedding)
                VALUES ($1, $2, $3, $4, $5, $6,
                        $7, $8, $9,
                        $10, $11, $12,
                        $13, $14, $15,
                        $16, $17,
                        $18, $19, $20)
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
            claim.text_embedding
        )
        return identifier

    async def insert_media_embedding(self, kind: str, media_id: int, embedding: list[float]):
        """Inserts a media embedding into the database."""
        query = """
                INSERT INTO media_embeddings (kind, media_id, embedding)
                VALUES ($1, $2, $3)
                ON CONFLICT (kind, media_id) DO UPDATE
                SET embedding = $3,
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
                    updated_at               = CURRENT_TIMESTAMP
                WHERE id = $21;
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

    async def insert(self, instance: VeritasBaseModel) -> int:
        """Inserts any (compatible) object into the database and returns the assigned ID."""
        match instance:
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
        assert instance.id is not None, "Object must have an ID."
        match instance:
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
                SELECT embedding FROM media_embeddings
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
            query = "SELECT media_id AS id, embedding FROM media_embeddings WHERE kind = $1"
            rows = await self._fetch(query, kind)
        return [(row["id"], row["embedding"]) for row in rows]

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
