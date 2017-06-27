PYINCLUDE = -I/usr/include/python2.5 \
 -I/usr/lib/python2.5/site-packages/numpy/core/include
CFLAGS = $(PYINCLUDE) -I../oifitslib -I/usr/local/include `pkg-config --cflags glib-2.0`
LIBS = -L../oifitslib -loifits -L/usr/local/lib -lcfitsio \
 `pkg-config --libs glib-2.0`

%.c:	%.pyx
	cython $<

%.o:	%.c
	gcc -c -fPIC $(CFLAGS) $<

%.so:	%.o
	gcc -shared $< -lm $(LIBS) -o $@


all:	nestbase.so lighthouse.so oinest.so oifits.so vmodel.so

nestbase.c: nestbase.pxd

lighthouse.c: lighthouse.pxd

oinest.c: oinest.pxd

clean:
	@rm -f *.c *.o *.so *~ core core.*
