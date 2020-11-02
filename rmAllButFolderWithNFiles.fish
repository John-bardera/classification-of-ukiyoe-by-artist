# !fish

set DATA_PATH ./data/
set DATA_N 100

for folder in (ls $DATA_PATH);
	if test (ls $DATA_PATH$folder | wc -l) -lt $DATA_N
		rm -rf $DATA_PATH$folder
	end
;end