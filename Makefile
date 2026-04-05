
ugpt: ugpt.cc Makefile
	g++ -std=c++20 $(shell pkgconf --cflags eigen3) -ggdb ugpt.cc -o ugpt

get-data:
	wget https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt

.PHONY: clean
clean:
	rm -rf ugpt ugpt.dSYM
