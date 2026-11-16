"""Command line of the campaign watcher (run in the background by the agent; its exit is the notification).

    python -m PAOFLOW.agentic.campaign.watch projects/<study>/campaign.py --interval 900
    python -m PAOFLOW.agentic.campaign.watch projects/<study>/campaign.py --once      # one poll, print, exit

The study's ``campaign.py`` defines ``CAMPAIGN``. Every poll rewrites ``STATUS.md``; findings already reported are
listed in ``.watch_reported`` of the study folder (delete it to see them all again). The library function is
:func:`~PAOFLOW.agentic.campaign.watcher.watch`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from PAOFLOW.agentic.campaign.watcher import load_campaign, watch


def main(argv: list[str] | None = None) -> int:
    """Command line: watch a study's campaign and print the event."""
    parser = argparse.ArgumentParser(
        description='Watch a campaign until something needs the agent.'
    )
    parser.add_argument('campaign_file', help="the study's campaign.py (defines CAMPAIGN)")
    parser.add_argument('--interval', type=float, default=900.0, help='seconds between polls')
    parser.add_argument('--max-hours', type=float, default=72.0)
    parser.add_argument('--stop-at', default='error,warning', help='severities that end the watch')
    parser.add_argument(
        '--once', action='store_true', help='poll once, print the status and the findings'
    )
    args = parser.parse_args(argv)
    campaign = load_campaign(args.campaign_file)
    event, findings = watch(
        campaign, args.interval, args.max_hours, tuple(args.stop_at.split(',')), once=args.once
    )
    print('CAMPAIGN %s: %s (%s)' % (campaign.name, event.upper(), time.strftime('%Y-%m-%d %H:%M')))
    for finding in findings:
        print(finding.line())
    print('status: %s' % os.path.join(campaign.study_dir, 'STATUS.md'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
