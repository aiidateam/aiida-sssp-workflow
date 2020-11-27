C-----------------------------------------------------------------------
      SUBROUTINE EFERMI (NEL,NBANDS,DEL,NSPPTS,NDIM8,NDIM10,
     +          WEIGHT,OCC,EF,EIGVAL,SORT,eigmin,eigmax,enocc,ismear)
C-----------------------------------------------------------------------

C     FERMI ENERGY & SMEARING PACKAGE REWRITTEN IN FINAL FORM FOR
C     CASTEP/CETEP USE BY A. DE VITA IN JULY 1992, STARTING FROM
C     R. NEEDS VERSION
C
C     COLD SMEARING, SPLINE OF GAUSSIANS, AND A CORRECTION TO METH-PAX
C     INTRODUCED BY N. MARZARI (1995)
C
C     FOR A DETAILED EXPLANATION OF THE SMEARING APPROACH, LOOK AT
C     N. MARZARI PhD THESIS, CHAPTER 4, AND REFERENCES THEREIN:
C     http://theossrv1.epfl.ch/uploads/Group/Bibliography/Nicola%20Marzari_these_04-1996.pdf
C
C     Cold smearing reference: N. Marzari, D. Vanderbilt, A. De Vita
C     and M. C. Payne, ``Thermal contraction and disordering of the
C     Al(110) surface'', Phys. Rev. Lett. 82, 3296 (1999).
C
C     GIVEN A SET OF WEIGHTS AND THE EIGENVALUES ASSOCIATED TO THEIR
C     ASSOCIATED K-POINTS FOR BZ SAMPLING, THIS SUBROUTINE PERFORMS
C     TWO TASKS:
C
C     (1) DETERMINES THE FERMI LEVEL AND THE OCCUPANCY OF THE STATES
C         ACCORDING TO CHOSEN SMEARING SCHEME

C     (2) DETERMINES THE "ENTROPY CORRECTION" TO BE ADDED TO
C         THE TOTAL ENERGY FOR RECOVERING THE ZERO-SMEARING ENERGY
C         (I.E. THE TRUE GROUND STATE ENERGY) AND PASSES IT BACK TO
C         THE MAIN CODE. NOTE THAT THIS IS NOT NEEDED FOR METH-PAX
C         OR THE COLD SMEARING
C
C     THE SMEARING SCHEMES (CHOOSE ONE WITH PARAMETER ISMEAR):
C
C     (1) GAUSSIAN:
C     SEE: C-L FU AND K-M HO, PHYS. REV. B 28, 5480 (1983).
C     THEIR IMPLEMENTATION WAS VARIATIONAL BUT *NOT* CORRECTED FOR
C     SECOND ORDER DEVIATION IN SIGMA AS ALSO WAS THE SIMILAR SCHEME
C     (WITH OPPOSITE SIGN DEVIATION) IN: R.J.NEEDS, R.M.MARTIN AND O.H.
C     NIELSEN, PHYS. REV. B 33 , 3778 (1986).
C     USING THE CORRECTION CALCULATED HEREAFTER EVERYTHING SHOULD BE OK.
C     THE SMEARING FUNCTION IS A GAUSSIAN NORMALISED TO 2.
C     THE OCCUPATION FUNCTION IS THE ASSOCIATED COMPLEMENTARY
C     ERROR FUNCTION.
C
C     (2) FERMI-DIRAC:
C     SEE: M.J.GILLAN J. PHYS. CONDENS. MATTER 1, 689 (1989), FOLLOWING
C     THE SCHEME OUTLINED IN J.CALLAWAY AND N.H.MARCH, SOLID STATE PHYS. 38,
C     136 (1984), AFTER D.N.MERMIN, PHYS. REV 137, A1441 (1965).
C     THE OCCUPATION FUNCTION IS TWICE THE SINGLE ELECTRON
C     FERMI-DIRAC DISTRIBUTION.
C
C     (3) HERMITE-DELTA_EXPANSION:
C     SEE: METHFESSEL AND PAXTON, PHYS. REV.B 40, 3616 (1989).
C     THE SMEARING FUNCTION IS A TRUNCATED EXPANSION OF DIRAC'S DELTA
C     IN HERMITE POLINOMIALS.
C     FOR THE SMEARING FUNCTION IMPLEMENTED HERE THE TRUNCATION IS
C     AT THE FIRST NON-TRIVIAL EXPANSION TERM D1(X).
C     THE OCCUPATION FUNCTION IS THE ASSOCIATED PRIMITIVE.
C     (NOTE: THE OCCUPATION FUNCTION IS NEITHER MONOTONIC NOR LIMITED
C     BETWEEN 0. AND 2. : PLEASE CHECK THE COMPATIBILITY OF THIS WITH
C     YOUR CODE'S VERSION AND VERIFY IN A TEST CALCULATION THAT THE
C     FERMI LEVEL IS *UNIQUELY* DETERMINED).
C
C     THE ENTROPY CORRECTION HOLDS UP TO THE THIRD ORDER IN DELTA AT LEAST,
C     AND IS NOT NECESSARY (PUT = 0.) FOR THE HERMITE_DELTA EXPANSION,
C     SINCE THE LINEAR ENTROPY TERM IN SIGMA IS ZERO BY CONSTRUCTION
C     IN THAT CASE. (well, we still need the correct free energy. hence
C     delcor is set to its true value, nmar)
C
C     (4) GAUSSIAN SPLINES:
C     similar to a Gaussian smearing, but does not require the
C     function inversion to calculate the gradients on the occupancies.
C     It is thus to be preferred in a scheme in which the occ are
C     independent variables (not in castep/cetep) (N. Marzari)
C
C     (5) POSITIVE HERMITE, or COLD SMEARING I
C     similar to Methfessel-Paxton (that is killing the linear order
C     in the entropy), but with positive-definite occupations (still
C     greater than 1) ! (N. Marzari)
C
C     (6) POSITIVE HERMITE, or COLD SMEARING II: the one to use.
C     (5) and (6) are practically identical; (6) is more elegant.
C     Phys. Rev. Lett. 82, 3296 (1999).

C------------------
C     **** PLEASE INQUIRE WITH ADV/NMAR FOR REFERENCE & SUGGESTIONS IF
C     YOU PLAN TO USE THE PRESENT CORRECTED BZ SAMPLING SCHEME ****
C     nicola.marzari@epfl.ch (was marzari@princeton.edu)
C     alessandro.de_vita@kcl.ac.uk (was alessandro.devita@epfl.ch)
C------------------

C     NEL ..... NUMBER OF ELECTRONS PER UNIT CELL
C     NBANDS .. NUMBER OF BANDS FOR EACH K-POINT
C     DEL ..... WIDTH OF GAUSSIAN SMEARING FUNCTION
C     NSPPTS .. NUMBER OF K-POINTS
C     NDIM8 ... MAXIMUM NUMBER OF BANDS AT A K-POINT
C     WEIGHT .. THE WEIGHT OF EACH K-POINT
C     OCC ..... THE OCCUPANCY OF EACH STATE
C     EF ...... THE FERMI ENERGY
C     SORT .... THE EIGENVALUES ARE WRITTEN INTO SORT WHICH IS
C               THEN SORTED INTO ASCENDING NUMERICAL VALUE, FROM
C               WHICH BOUNDS ON EF CAN EASILY BE OBTAINED
C     EIGVAL .. CONTAINS THE BEST EIGENVALUES AVAILABLE
c
c    ISMEAR = 1 GAUSSIAN BROADENING
C           = 2 FERMI-DIRAC BROADENING
C           = 3 HERMITE EXPANSION (1ST ORD.) (right delcor now, nmar)
C           = 4 SPLINE OF GAUSSIANS (nmar)
C           = 5 COLD SMEARING I (nmar)
C           = 6 COLD SMEARING II (nmar)
C     JMAX      THE MAX NUMBER OF BISECTIONS TO GET EF
C     XACC      THE DESIRED ACCURACY ON EF
C
C     DELCOR    THE CORRECTION. PASSED BACK IN SORT(1) (= SORT(1,1))
C               TO BE COMPATIBLE WITH OLD VERSIONS OF THE CODE. IF
C               YOUR CODE DOES NOT DO IT ALREADY, ADD SORT(1) TO
C               CORRECT THE TOTAL ENERGY IN YOUR MAIN.FORTRAN , WHICH
C               SHOULD CALL ANYWAY THIS ROUTINE CORRECTLY.
C
C     CHANGED: -TS goes into enocc (nmar)
C     also: the correction is needed for 1,2 and 4 only
C-------------------------------------------------------------
c     ANOTHER NOTE:
C     Thanks to the possible > 2 or < 0
c     orbital occupancies in the general case of smearing function,
c     (e.g. in the M-P case) the algorithm to find EF has been
c     chosen to be the robust bisection method (from Numerical
c     Recipes) to allow for non monotonic relation between total
c     NEL (see above) and EF. One value for EF which solves
c     NEL(EF) - Z = 0   is always found.
C---------------------------------------------------------------------
C
      IMPLICIT REAL (A-H,O-Z)

      DIMENSION SORT(NDIM8*NDIM10)
      DIMENSION OCC(NDIM8,NDIM10),WEIGHT(NDIM10),EIGVAL(NDIM8,NDIM10)
      EXTERNAL ERF_C
      EXTERNAL FERMID
      EXTERNAL DELTHM
      EXTERNAL POSHM
      EXTERNAL POSHM2
C.....WARNINGS
      COMMON /WARN/ IWARN
      PARAMETER ( JMAX =200, XACC=1.0D-10)
C.....WHICH SMEARING IS USED

      pi=acos(0.0)*2.0
      ee=exp(1.0)
      eesh=sqrt(ee)*0.5
      sq2i=sqrt(2.0)*0.5
      piesqq=sqrt(ee*pi)*0.25

      Z    = FLOAT(NEL)

C COPY EIGVAL INTO SORT ARRAY.

      NEIG = 0
      DO 10 ISPPT = 1,NSPPTS
        DO 20  J = 1, NBANDS
          NEIG = NEIG + 1
20        SORT(NEIG) = EIGVAL(J,ISPPT)
10      CONTINUE

C=======================================================================
C  THE ARRAY IS ORDERED INTO ASCENDING ORDER OF EIGENVALUE
C=======================================================================

      DO 26 N=2,NSPPTS*NBANDS
        EN=SORT(N)
        DO 22 NN=N-1,1,-1
          IF (SORT(NN).LE.EN) GO TO 24
          SORT(NN+1)=SORT(NN)
  22    CONTINUE
        NN=0
  24    SORT(NN+1)=EN
  26  CONTINUE
      eigmin=sort(1)
      eigmax=sort(nsppts*nbands)

C===========================
C     THE UPPER BOUND XE2 AND THE LOWER BOUND XE1
C     ARE PUT TO FIRST AND LAST EIGENVALUE, THEN
C     THE ACTUAL FERMI ENERGY IS FOUND BY BISECTION
C===========================

      XE1=ef-del
      XE2=ef+0.5*del
      xe0=ef

C
c           WRITE(*,*) ' '
          IF(ISMEAR.EQ.1) THEN
            WRITE(*,*) 'GAUSSIAN BROADENING'
          ELSEIF(ISMEAR.EQ.2) THEN
            WRITE(*,*) 'FERMI-DIRAC BROADENING'
          ELSEIF(ISMEAR.EQ.3) THEN
            WRITE(*,*) 'HERMITE-DIRAC BROADENING'
          ELSEIF(ISMEAR.EQ.4) THEN
            WRITE(*,*) 'GAUSSIAN SPLINES BROADENING'
          ELSEIF(ISMEAR.EQ.5) THEN
            WRITE(*,*) 'COLD SMEARING I'
          ELSEIF(ISMEAR.EQ.6) THEN
            WRITE(*,*) 'COLD SMEARING II'
          ENDIF
c     WRITE (9,*) '                  '
c     WRITE (9,*) 'min & max eigval ',eigmin,eigmax

      nmax=1
100   continue

C
C FMID = FUNC(X2) in Numerical Recipes.
C
      Z1=0.D0
      DO 40 ISPPT = 1,NSPPTS
        DO 50 J = 1,NBANDS
          X = (XE2 - EIGVAL(J,ISPPT))/DEL
          IF(ISMEAR.EQ.1) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*( 2.0 - ERF_C(X) )
          ELSEIF(ISMEAR.EQ.2) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*FERMID(-X)
          ELSEIF(ISMEAR.EQ.3) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*DELTHM(X)
          ELSEIF(ISMEAR.EQ.4) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*SPLINE(-X)
          ELSEIF(ISMEAR.EQ.5) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*POSHM(X)
          ELSEIF(ISMEAR.EQ.6) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*POSHM2(X)
          ENDIF
50      CONTINUE
40    CONTINUE

      FMID= Z1-Z

C F = FUNC(X1)

      Z1=0.D0
      DO 140 ISPPT = 1,NSPPTS
        DO 150 J = 1,NBANDS
          X = (XE1 - EIGVAL(J,ISPPT))/DEL
          IF(ISMEAR.EQ.1) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*( 2.0 - ERF_C(X) )
          ELSEIF(ISMEAR.EQ.2) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*FERMID(-X)
          ELSEIF(ISMEAR.EQ.3) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*DELTHM(X)
          ELSEIF(ISMEAR.EQ.4) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*SPLINE(-X)
          ELSEIF(ISMEAR.EQ.5) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*POSHM(X)
          ELSEIF(ISMEAR.EQ.6) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*POSHM2(X)
          ENDIF
150     CONTINUE
140   CONTINUE

      F= Z1-Z

      IF(F*FMID .GE. 0.D0) THEN
        if (nmax.ge.10000) then
         WRITE(*,*) 'ERROR: NO FERMI ENERGY FOUND ',
     &             xe1,xe2,f,fmid,nmax
         WRITE(*,*) ' '
         WRITE(*,*) 'IS THE ELECTRONIC TEMPERATURE TOO SMALL ? ',del
         WRITE(*,*) ' '
         stop
        else
         nmax=nmax+1
         XE1=XE0-float(nmax)*del
         XE2=XE0+(float(nmax)-0.5)*del
         goto 100
        end if
      ENDIF
      IF(F .LT. 0.D0) THEN
       RTBIS = XE1
       DX = XE2 - XE1
      ELSE
       RTBIS = XE2
       DX = XE1 - XE2
      ENDIF
      DO 42 J = 1, JMAX
       DX = DX * 0.5D0
       XMID = RTBIS + DX

C FMID=FUNC(XMID)

      Z1=0.D0
      DO 240 ISPPT = 1,NSPPTS
        DO 250 J2 = 1,NBANDS
          X = (XMID - EIGVAL(J2,ISPPT))/DEL
          IF(ISMEAR.EQ.1) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*( 2.0 - ERF_C(X) )
          ELSEIF(ISMEAR.EQ.2) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*FERMID(-X)
          ELSEIF(ISMEAR.EQ.3) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*DELTHM(X)
          ELSEIF(ISMEAR.EQ.4) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*SPLINE(-X)
          ELSEIF(ISMEAR.EQ.5) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*POSHM(X)
          ELSEIF(ISMEAR.EQ.6) THEN
            Z1 = Z1 + WEIGHT(ISPPT)*POSHM2(X)
          ENDIF
250     CONTINUE
240   CONTINUE

      FMID= Z1-Z

      IF(FMID .LE. 0.D0) RTBIS=XMID
      IF(ABS(DX) .LT. XACC .OR. FMID .EQ. 0) THEN
       GOTO 1042
      ENDIF
  42  CONTINUE
        WRITE(*,*) 'CANNOT BISECT FOREVER, CAN I ?'
c       WRITE(9,*) 'CANNOT BISECT FOREVER, CAN I ?'
        STOP

1042  EF = RTBIS

170   CONTINUE

      WRITE (*,*) EF
c     WRITE (9,180) EF
180   FORMAT (' FERMI ENERGY = ',F15.8,' EV')

C     FORM OCCUPATIONS OCC(NBDS,NSPPTS)
      DO 190 ISPPT = 1, NSPPTS
        DO 200 J = 1,NBANDS
          X = ( EF-EIGVAL(J,ISPPT))/DEL
          IF(ISMEAR.EQ.1) THEN
            OCC(J,ISPPT) = 2.0 - ERF_C(X)
          ELSEIF(ISMEAR.EQ.2) THEN
            OCC(J,ISPPT) = FERMID(-X)
          ELSEIF(ISMEAR.EQ.3) THEN
            OCC(J,ISPPT) = DELTHM(X)
          ELSEIF(ISMEAR.EQ.4) THEN
            OCC(J,ISPPT) = SPLINE(-X)
          ELSEIF(ISMEAR.EQ.5) THEN
            OCC(J,ISPPT) = POSHM(X)
          ELSEIF(ISMEAR.EQ.6) THEN
            OCC(J,ISPPT) = POSHM2(X)
          ENDIF
            OCC(J,ISPPT) = OCC(J,ISPPT)/2.0
200     CONTINUE
190   CONTINUE

C-------------------------------------------------------------
C CALCULATES THE CORRECTION TERM TO GET "0 TEMPERATURE" ENERGY
C-------------------------------------------------------------

      DELCOR=0.0D0
      DO 191 ISPPT = 1, NSPPTS
       DO 201 J = 1,NBANDS
        X = ( EF-EIGVAL(J,ISPPT))/DEL
        IF(ISMEAR.EQ.1) THEN
         DELCOR=DELCOR
     &    -DEL*WEIGHT(ISPPT)*EXP(-X*X)/(2.D0*SQRT(pi))
        ELSEIF(ISMEAR.EQ.2) THEN
         FI=FERMID(-X)/2.D0
         IF(ABS(FI) .GT. 1.E-06) THEN
          IF(ABS(FI-1.D0) .GT. 1.E-06) THEN
           DELCOR=DELCOR+DEL*WEIGHT(ISPPT)*
     &     (FI*LOG(FI)+(1.D0-FI)*LOG(1.D0-FI))
          ENDIF
         ENDIF
        ELSEIF(ISMEAR.EQ.3) THEN
         DELCOR=DELCOR+DEL/2.0*WEIGHT(ISPPT)
     &    *(2.0*x*x-1)*exp(-x*x)/(2.0*sqrt(pi))
        ELSEIF(ISMEAR.EQ.4) THEN
         x=abs(x)
         zeta=eesh*abs(x)*exp(-(x+sq2i)**2)+piesqq*ERF_C(x+sq2i)
         delcor=delcor-del*WEIGHT(ISPPT)*zeta
        ELSEIF(ISMEAR.EQ.5) THEN
         a=-0.5634
c        a=-0.8165
         DELCOR=DELCOR-DEL/2.0*WEIGHT(ISPPT)
c NOTE g's are all intended to be normalized to 1 !
c this following line is -2*int_minf^x [t*g(t)]dt
     &   *(2.0*a*x**3-2.0*x*x+1 )*exp(-x*x)/(2.0*sqrt(pi))
        ELSEIF(ISMEAR.EQ.6) THEN
         DELCOR=DELCOR-DEL/2.0*WEIGHT(ISPPT)
c NOTE g's are all intended to be normalized to 1 !
c this following line is -2*int_minf^x [t*g(t)]dt
     &   *(1-sqrt(2.0)*x)*exp(-(x-1/sqrt(2.))**2)/sqrt(pi)
        ENDIF
201    CONTINUE
191   CONTINUE

c--------------------------------------------------------
C  the correction stored in enocc
c--------------------------------------------------------

      sort(1)=delcor
      enocc=2.0*delcor

c--------------------------------------------------------
c     TEST WHETHER OCCUPANCY ADDS UP TO Z
c--------------------------------------------------------

      TEST = 0.0
      DO 210 ISPPT = 1,NSPPTS
        DO 215 J = 1,NBANDS
215       TEST = TEST + WEIGHT(ISPPT)*OCC(J,ISPPT)
210     CONTINUE
      test=test*2.0

c--------------------------------------------------------
c UNCOMMENT IF YOU WISH to see occupation numbers at each kpoint
c--------------------------------------------------------

      DO 193 ISPPT = 1, NSPPTS

c         WRITE (*,9897) ISPPT
c         WRITE (9,9897) ISPPT
9897  FORMAT(' OCCUPATION NUMBERS AT K-POINT',I3,':')
c         WRITE(*,9898) (OCC(J,ISPPT),J=1,NBANDS)
c         WRITE(9,9898) (OCC(J,ISPPT),J=1,NBANDS)
9898  FORMAT(2X,5F15.6)
193     CONTINUE

      IF ( ABS(TEST-Z) .GT. 1.0D-5) THEN

C        WRITE(*,*) '*** ERROR ***'
        WRITE(*,*) '*** Warning ***'
        WRITE(*,*) TEST,NEL
220     FORMAT(' SUM OF OCCUPANCIES =',F18.12 ,' BUT NEL =',I5)
C        STOP     !! Modified by A. Marrazzo and G. Prandini
C		  !! to do bands convergence tests.
C

      ELSE

c       WRITE(*,230) TEST
c       WRITE(*,*) ' '
230     FORMAT(' TOTAL CHARGE = ',F15.8)

      ENDIF
C
C     TEST WHETHER THE MATERIAL IS A SEMICONDUCTOR
C     (ASSUMING SPIN DEGENERACY)
C
C     this has to be sorted out better
C
      IF ( MOD( NEL, 2) .EQ. 1) RETURN
      INEL = NEL/2
      if (inel+1.le.ndim8) then
       ELOW = EIGVAL(INEL+1,1)
       DO 310 ISPPT = 2,NSPPTS
         ELOW =MIN( ELOW, EIGVAL(INEL+1,ISPPT))
310    CONTINUE
       DO 320 ISPPT = 1,NSPPTS
        IF (ELOW .LT. EIGVAL(INEL,ISPPT)) RETURN
320    CONTINUE
      end if

      if (NSPPTS.gt.1) then
       WRITE (*,*) 'MATERIAL MAY BE A SEMICONDUCTOR'
      end if
C
      RETURN
      END
C
C =========================================================================
C
      FUNCTION ERF_C(XX)
C
C     COMPLEMENTARY ERROR FUNCTION
C     FROM THE SANDIA MATHEMATICAL PROGRAM LIBRARY
C
C     XMAX IS THE VALUE BEYOND WHICH ERF_C(X) = 0 .
C     IT IS COMPUTED AS SQRT(LOG(RMIN)), WHERE RMIN IS THE
C     SMALLEST REAL NUMBER REPRESENTABLE ON THE MACHINE.
C     IBM VALUE: (THE INTRINSIC ERF_C COULD ALSO BE USED)
C     PARAMETER ( XMAX = 13.4 )
C     VAX VALUE: (XMAX = 9.3)
C -----------------------------------
C     12-Mar-90  Obtained from B. Hammer
C     12-MAR-90  Changed to single precision at the end XW
C                also XX1
      PARAMETER ( XMAX = 9.3D0)
C
C      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION P1(4),Q1(4),P2(6),Q2(6),P3(4),Q3(4)
C
      DATA P1 /242.66 79552 30531 8D0 , 21.979 26161 82941 5D0 ,
     + 6.9963 83488 61913 6D0 , -3.5609 84370 18153 9D-2/
      DATA Q1 /215.05 88758 69861 2D0 , 91.164 90540 45149 0D0,
     + 15.082 79763 04077 9D0 , 1.0D0/
      DATA P2 /22.898 99285 1659D0 , 26.094 74695 6075D0 ,
     + 14.571 89859 6926D0 , 4.2677 20107 0898D0 ,
     + 0.56437 16068 6381D0 , -6.0858 15195 9688 D-6/
      DATA Q2 /22.898 98574 9891D0 , 51.933 57068 7552D0 ,
     + 50.273 20286 3803D0 , 26.288 79575 8761D0 ,
     + 7.5688 48229 3618D0 , 1.0D0/
      DATA P3 /-1.2130 82763 89978 D-2 , -0.11990 39552 68146 0D0 ,
     + -0.24391 10294 88626D0 , -3.2431 95192 77746 D-2/
      DATA Q3 /4.3002 66434 52770 D-2 , 0.48955 24419 61437D0 ,
     + 1.4377 12279 37118D0 , 1.0D0/
C     1/SQRT(PI)
      DATA SQPI /0.56418 95835 47756D0/
C
C----------------------------------------------------------------------
      IF (XX .GT.  XMAX)    GOTO 330
      IF (XX .LT. -XMAX)    GOTO 320
      X = ABS(XX)
      X2 = X*X
      IF (X .GT. 4.0D0)     GOTO 300
      IF (X .GT. 0.46875D0) GOTO 200
C
C     -46875 < X < 0.46875
      ERF_C = X*(P1(1) + X2*(P1(2) + X2*(P1(3) + X2*P1(4))))
      ERF_C = ERF_C/(Q1(1) + X2*(Q1(2) + X2*(Q1(3) + X2*Q1(4))))
      IF (XX .LT. 0.0) ERF_C = - ERF_C
      ERF_C = 1.0D0 - ERF_C
      GOTO 9999
C
200   ERF_C = EXP( -X2)*(P2(1) + X*(P2(2) + X*(P2(3) + X*(P2(4) +
     + X*(P2(5) + X*P2(6))))))
      ERF_C = ERF_C/(Q2(1) + X*(Q2(2) + X*(Q2(3) + X*(Q2(4) + X*(Q2(5) +
     + X*Q2(6))))))
      IF (XX .LE. 0.0) ERF_C = 2.0D0 - ERF_C
      GOTO 9999
C
300   XI2 = 1.0D0/X2
      ERF_C = XI2*(P3(1) + XI2*(P3(2) + XI2*(P3(3) + XI2*P3(4))))/
     + (Q3(1) + XI2*(Q3(2) + XI2*(Q3(3) + XI2*Q3(4))))
      ERF_C = EXP( -X2)*(SQPI + ERF_C)/X
      IF (XX .LT. 0.0) ERF_C = 2.0D0 - ERF_C
      GOTO 9999
C
320   ERF_C = 2.0D0
      GOTO 9999
330   ERF_C = 0.0D0
C
9999  RETURN
      END
C================================================================
      FUNCTION FERMID(XX)
C
      IF(XX .GT. 30.D0) THEN
        FERMID=0.D0
      ELSEIF(XX .LT. -30.D0) THEN
        FERMID=2.D0
      ELSE
        FERMID=2.D0/(1.D0+EXP(XX))
      ENDIF
C
      RETURN
      END
C================================================================
      FUNCTION DELTHM(XX)
C
      pi=3.14159265358979
      IF(XX .GT. 10.D0) THEN
        DELTHM=2.D0
      ELSEIF(XX .LT. -10.D0) THEN
        DELTHM=0.D0
      ELSE
        DELTHM=(2.D0-ERF_C(XX))+XX*EXP(-XX*XX)/SQRT(PI)
      ENDIF
C
      RETURN
      END
C================================================================
      FUNCTION SPLINE(X)

      eesqh=sqrt(exp(1.0))*0.5
      sq2i=sqrt(2.0)*0.5
      if (x.ge.0.0) then
        fx=eesqh*exp(-(x+sq2i)**2)
      else
        fx=1.0-eesqh*exp(-(x-sq2i)**2)
      endif
      spline=2.0*fx
c
      return
      end
C================================================================
      FUNCTION POSHM(X)
C
c NOTE g's are all intended to be normalized to 1 !
C function = 2 * int_minf^x [g(t)] dt
C
      pi=3.141592653589793238
      a=-0.5634
c     a=-0.8165
      IF(X .GT. 10.D0) THEN
        POSHM=2.D0
      ELSEIF(X .LT. -10.D0) THEN
        POSHM=0.D0
      ELSE
        POSHM=(2.D0-ERF_C(X))+(-2.0*a*x*x+2*x+a)*EXP(-X*X)/SQRT(PI)/2.0
      ENDIF
C
      RETURN
      END
C================================================================
      FUNCTION POSHM2(X)
C
c NOTE g's are all intended to be normalized to 1 !
C function = 2 * int_minf^x [g(t)] dt
C
      pi=3.141592653589793238
      IF(X .GT. 10.D0) THEN
        POSHM2=2.D0
      ELSEIF(X .LT. -10.D0) THEN
        POSHM2=0.D0
      ELSE
        POSHM2=(2.D0-ERF_C(X-1./sqrt(2.)))+
     &  sqrt(2.)*exp(-x*x+sqrt(2.)*x-0.5)/sqrt(pi)
      ENDIF
C
      RETURN
      END
c===========================WELLWELLWEHAVEFINISHED.