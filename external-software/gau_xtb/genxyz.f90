!This tool was written by Dr. Tian Lu at Beijing Kein Research Center for Natural Sciences (www.keinsci.com)
!Contact: sobereva@sina.com

program genxyz
implicit real*8 (a-h,o-z)
character*2 :: ind2name(0:150)=(/ "Bq","H ","He", &   !X(number O) is ghost atom
"Li","Be","B ","C ","N ","O ","F ","Ne", & !3~10
"Na","Mg","Al","Si","P ","S ","Cl","Ar", & !11~18
"K ","Ca","Sc","Ti","V ","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr", & !19~36
"Rb","Sr","Y ","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I ","Xe", & !37~54
"Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu", & !55~71
"Hf","Ta","W ","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn", & !72~86
"Fr","Ra","Ac","Th","Pa","U ","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr", & !87~103
"Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Ut","Fl","Up","Lv","Us","Uo","Un","Ux",("??",i=121,150) /)

b2a=0.529177249D0

open(10,file="mol.tmp",status="old")
open(11,file="mol.xyz",status="replace")
read(10,*) natm
write(11,*) natm
read(10,*)
write(11,*)
do i=1,natm
	read(10,*) idx,x,y,z
	write(11,*) ind2name(idx),x*b2a,y*b2a,z*b2a
end do
close(10)
close(11)
end program

