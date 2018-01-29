clean:
	rm -rf ../cpp/*

upload-pi:
	rm -rf ../cpp/*
	python3 generate.py
	scp ../cpp/* pi@129.217.34.97:/home/pi/tree-source 
bare-run:
	rm -rf ../cpp/*
	python3 generate.py
