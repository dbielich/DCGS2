
include ../make.inc

all: xmpi_main_qr_cgs.exe xmpi_main_qr_cgs2.exe xmpi_main_qr_dcgs2.exe xmpi_main_qr_hh_lvl1.exe xmpi_main_qr_hh_lvl2.exe xmpi_main_qr_mgs_lvl1.exe xmpi_main_qr_mgs_lvl2.exe

xmpi_main_qr_cgs.exe:   mpi_main_qr_cgs.o ../orth_scheme/mpi_orth_cgs_twosynch_append1.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o
	$(LOADER) -o $@ mpi_main_qr_cgs.o ../orth_scheme/mpi_orth_cgs_twosynch_append1.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o $(LIB)

xmpi_main_qr_cgs2.exe:  mpi_main_qr_cgs2.o ../orth_scheme/mpi_orth_cgs2_threesynch_append1.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o
	$(LOADER) -o $@ mpi_main_qr_cgs2.o ../orth_scheme/mpi_orth_cgs2_threesynch_append1.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o $(LIB)

xmpi_main_qr_dcgs2.exe: mpi_main_qr_dcgs2.o ../orth_scheme/mpi_orth_dcgs2_onesynch_append1_qr.o ../orth_scheme/mpi_orth_dcgs2_onesynch_cleanup_append1_qr.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o 
	$(LOADER) -o $@ mpi_main_qr_dcgs2.o ../orth_scheme/mpi_orth_dcgs2_onesynch_append1_qr.o ../orth_scheme/mpi_orth_dcgs2_onesynch_cleanup_append1_qr.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o $(LIB)

xmpi_main_qr_hh_lvl1.exe: mpi_main_qr_hh_lvl1.o ../orth_scheme/mpi_orth_hh_lvl1_manysynch_append1.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o 
	$(LOADER) -o $@   mpi_main_qr_hh_lvl1.o ../orth_scheme/mpi_orth_hh_lvl1_manysynch_append1.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o $(LIB)

xmpi_main_qr_hh_lvl2.exe: mpi_main_qr_hh_lvl2.o ../orth_scheme/mpi_orth_hh_lvl2_fivesynch_append1.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o 
	$(LOADER) -o $@   mpi_main_qr_hh_lvl2.o ../orth_scheme/mpi_orth_hh_lvl2_fivesynch_append1.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o $(LIB)

xmpi_main_qr_mgs_lvl1.exe: mpi_main_qr_mgs_lvl1.o ../orth_scheme/mpi_orth_mgs_lvl1_manysynch_append1.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o 
	$(LOADER) -o $@    mpi_main_qr_mgs_lvl1.o ../orth_scheme/mpi_orth_mgs_lvl1_manysynch_append1.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o $(LIB)

xmpi_main_qr_mgs_lvl2.exe: mpi_main_qr_mgs_lvl2.o ../orth_scheme/mpi_orth_mgs_lvl2_threesynch_append1.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o 
	$(LOADER) -o $@    mpi_main_qr_mgs_lvl2.o ../orth_scheme/mpi_orth_mgs_lvl2_threesynch_append1.o ../checks/mpi_check_qq_orth.o ../checks/mpi_check_qr_repres.o $(LIB)

.c.o:
	$(CC) $(OPTS) -I$(INCLUDE) -c -o $@ $<

clean:
	rm -f *o *exe ../checks/*o ../checks/*exe ../orth_scheme/*o ../orth_scheme/*exe
