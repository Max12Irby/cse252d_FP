#!/bin/bash

# Act like regular kubesh so code doesn't break
cd /opt/launch-sh/bin/

# Load libraries
BASEDIR=${BASEDIR:-$(dirname $(cd $(dirname ${BASH_SOURCE[0]}) && pwd))}
LIBDIR=${BASEDIR}/lib

#########
# Set up our core K8S environment vars (username, UID, kubectl location)
source "${LIBDIR}/kubevars.sh"

POD_NAME=$1
shift
if [ \! "$POD_NAME" ]; then
	( echo "Usage: $0 <pod-name> [cmd]"; echo "       (if [cmd] is omitted, an interactive Bash shell is started)" ) 1>&2 && exit 1
fi

if ! kubectl get pod $POD_NAME > /dev/null 2>&1 ; then
	echo "Error: Pod \"$POD_NAME\" does not appear to exist. Run 'kubectl get pods' to list running pods." 1>&2 && exit 1
fi

INITENV=$(kubectl get pod $POD_NAME -o jsonpath='{.spec.containers[0].command[2]}')

if [ $# == 0 ]; then
	set -- /bin/bash
fi

exec_flags="-c c1"
if [ "$KUBESH_NO_TTY" ]; then
	exec_flags=""
fi

kubectl exec ${exec_flags} ${POD_NAME} -- ${INITENV} "$@"
