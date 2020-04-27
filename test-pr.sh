#!/bin/bash

PULL=${1}
REMOTE_BRANCH=refs/pull/${PULL}/head
LOCAL_BRANCH=pull/pr${PULL}
LOCAL_TEST_BRANCH=test-pr${PULL}

echo "Fetching origin/${REMOTE_BRANCH} as ${LOCAL_BRANCH}"
git fetch -n origin ${REMOTE_BRANCH}:${LOCAL_BRANCH}
echo "Merging ${LOCAL_BRANCH} into ${LOCAL_TEST_BRANCH} on top of master"
git checkout -b ${LOCAL_TEST_BRANCH} master
git merge ${LOCAL_BRANCH}
