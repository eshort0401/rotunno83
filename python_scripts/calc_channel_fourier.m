syms zet zetap L zetaT N1 N2 k R zetaT 
assume(zet,'real')
assume(zet,'positive')
assume(k,'real')
assume(k,'positive')
assume(R,'real')
assume(R,'positive')
assume(zetaT,'real')
assume(zetaT,'positive')
test = (-exp(-zet)*(sin(k*zet)+k*cos(k*zet))+k)/(k^2+1)*exp(1i*k/R*zet)*exp(-1i*zetaT*(k/R+k)) - (1i*k-1)/(k^2+1)*exp((-1i*k-1)*zet)*sin(k*zetaT)*exp(1i*k/R*zetaT)*exp(-1i*k/R*zet)
expand(test)