import kaggle
import json
from download_results import authenticate
from kagglesdk.competitions.types.submission_status import SubmissionStatus

api = authenticate()


def get_latest_submission_score(competition_name):
    try:
        # Fetch the list of submissions for the authenticated user
        submissions = api.competition_submissions(competition_name)
        # print(submissions[0])
        if not submissions:
            print("No submissions found.")
            return Non
        # Sort submissions by date, latest first
        latest_submission = submissions[0]
        # print(latest_submission, type(latest_submission))
        # print(latest_submission.__dict__, type(latest_submission.__dict__))
        # print(help(latest_submission))
        print(latest_submission.__dict__.keys())
        print(latest_submission.__dict__.get('_status'))
        print(latest_submission.status, type(latest_submission._status))
        # print(help(latest_submission))
        # print(help(SubmissionStatus))
        assert latest_submission._status == SubmissionStatus.COMPLETE
        return latest_submission.public_score

    except Exception as e:
        print("Error fetching submissions:", e)
        return None


# Usage
competition_name = "surveillance-for-retail-stores"
latest_score = get_latest_submission_score(competition_name)
if latest_score is not None:
    print(f"Latest Submission Public Score: {latest_score}")
