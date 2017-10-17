#!/bin/bash

CURRENT_DIR=`pwd`
WORKING_DIR=~/CWBQuery
TEMPDIR=$WORKING_DIR/tempDir
START_TIME="2005/01/01 00:00:00"
BEGIN_TIME_VAR_FILE=/tmp/beginTime.txt
SEISCOMP3_ARCHIVE=/opt/seiscomp3/var/lib/archive
CWB_QUERY_JAR_FILE=CWBQuery.jar
CWB_QUERY_SERVER_IP=13.55.154.202
TARBALL_FILE=EdgeCWBRelease.tar.gz

if [ ! -d $WORKING_DIR ]; then
	mkdir $WORKING_DIR
fi

if ! type -p java; then
	echo "This is the first time this script is running ..."
	echo "Installing java ... "
	sudo yum install java-1.7.0-openjdk -y
fi

if [ ! -f $BEGIN_TIME_VAR_FILE ]; then
	echo "Initializing the BEGIN_TIME VARIABLE"
	echo $START_TIME > $BEGIN_TIME_VAR_FILE
fi

if ! type -p scart ; then
	echo "Seiscomp3 is not installed or there is some problem with the scart tool! Aborting."
	exit 1
fi

cd $WORKING_DIR

JARS=`ls *.jar | xargs`
if [ -z "${JARS// }"]; then
	echo "Jars have not been downloaded yet. Downloading ..."
	mkdir $TEMPDIR
	cd $TEMPDIR

	if [ ! -f  $TARBALL_FILE ]; then
	        wget ftp://hazards.cr.usgs.gov/CWBQuery/$TARBALL_FILE
	        tar -xzvf $TARBALL_FILE
	        cp `find . -name *.jar | xargs` $WORKING_DIR
	fi

	cd $WORKING_DIR
	rm -rf $TEMPDIR
fi

BEGIN_TIME=`cat $BEGIN_TIME_VAR_FILE`
echo "BEGIN_TIME = $BEGIN_TIME"

CMD="java -jar -Xmx1600m $CWB_QUERY_JAR_FILE -h $CWB_QUERY_SERVER_IP -t ms -s '............' -b '$BEGIN_TIME' -d 3600"
echo "Executing the command $CMD ... "
eval $CMD

for ms_file in `ls *.ms | xargs`; do
	if [ -f ${ms_file} ]; then
		scart -I ${ms_file}
		rm *.ms
	fi
done

echo "Finished ingesting the miniseeds for the timewindow $BEGIN_TIME + 1 hour"

NEXT_BEGIN_TIME=$(date '+%Y/%m/%d %H:%M:%S' -d "$BEGIN_TIME 1 hour")
echo "Next begin time = $NEXT_BEGIN_TIME "
echo $NEXT_BEGIN_TIME > $BEGIN_TIME_VAR_FILE

cd $CURRENT_DIR
