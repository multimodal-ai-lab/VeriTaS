from veritas.db import db


async def print_review_dismissed_reasons():
    """Prints all unique dismissed reasons."""
    await db.connect_maybe_initialize()
    reasons = await db.get_dismissed_reasons()
    for reason, count in reasons.most_common():
        print(f"{count}: {reason}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(print_review_dismissed_reasons())
