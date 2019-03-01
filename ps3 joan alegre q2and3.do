/////////////////////
/////QUESTION 02/////
/////////////////////
clear all
cd "C:\Users\Joana\Desktop\Cole\18-19\2.Development\ps3\boyao"
use "ps03_mainq01.dta", replace
///Q2A: INCOME QUINTILE///

sort newid02 time
bysort newid02: egen mean_ln_y = mean(lny)
xtile income_quintile = mean_ln_y if model01 == 1 , nquantiles(5)
forvalues q=1/5	{
	quietly estpost sum beta01_i beta02_i pi01_i pi02_i if income_quintile==`q' & time==2, detail
	est store q02a_`q'
}

esttab q02a_*, replace label  ///
cells("mean(pattern(1 1 1 1 1) fmt(3)) p50(pattern(1 1 1 1 1) fmt(3))") width(\hsize) ///
mtitles("Q1" "Q2" "Q3" "Q4" "Q5") title("Coefficients by Income Quintile"\label{q02a})

esttab q02a_* using "q02a.tex", replace label  ///
cells("mean(pattern(1 1 1 1 1) fmt(3)) p50(pattern(1 1 1 1 1) fmt(3))") width(\hsize) ///
mtitles("Q1" "Q2" "Q3" "Q4" "Q5") title("Coefficients by Income Quintile"\label{q02a})

preserve
table income_quintile if time==2, contents(mean beta01_i p50 beta01_i mean beta02_i p50 beta02_i )  replace 
list   
twoway (line table1  income_quintile) (line table2  income_quintile) 	///
(line table3  income_quintile) (line table4  income_quintile)
graph export "q02a_coef.png", as(png) replace
restore
		
		//Q2A: percapita income quintile: is the previous result driven by household size
gen lny_pc = lny/log(familysize)
sort newid02 time
bysort newid02: egen mean_ln_y_pc = mean(lny_pc)
xtile incomepc_quintile = mean_ln_y_pc if model01 == 1 , nquantiles(5)
forvalues q=1/5	{
	quietly estpost sum beta01_i beta02_i pi01_i pi02_i if incomepc_quintile==`q' & time==2, detail
	est store q02a_`q'
}

preserve
table incomepc_quintile if time==2, contents(mean beta01_i p50 beta01_i mean beta02_i p50 beta02_i )  replace 
list   
twoway (line table1  incomepc_quintile) (line table2  incomepc_quintile) 	///
(line table3  incomepc_quintile) (line table4  incomepc_quintile)
graph export "q02a_percap.png", as(png) replace
restore


///Q2B: WEALTH QUINTILE///
	//Merge wealth variable of 2010 from my Problem Set1 since we have no wealth data here
rename hh HHID
merge m:1 HHID using "hh_wealth_2010.dta", keepusing(hh_wealth_2010)	//merge result looks good
drop if _merge==2
drop _merge
xtile wealth_quintile = hh_wealth_2010 if model01 == 1, nquantiles(5)

forvalues q=1/5	{
	quietly estpost sum beta01_i beta02_i pi01_i pi02_i if wealth_quintile==`q' & time==2, detail
	est store q02b_`q'
}

esttab q02b_*, replace label  ///
cells("mean(pattern(1 1 1 1 1) fmt(3)) p50(pattern(1 1 1 1 1) fmt(3))") width(\hsize) ///
mtitles("Q1" "Q2" "Q3" "Q4" "Q5") title("Coefficients by Wealth Quintile"\label{q02b})

esttab q02b_* using "q02b.tex", replace label  ///
cells("mean(pattern(1 1 1 1 1) fmt(3)) p50(pattern(1 1 1 1 1) fmt(3))") width(\hsize) ///
mtitles("Q1" "Q2" "Q3" "Q4" "Q5") title("Coefficients by Wealth Quintile"\label{q02b})

preserve
table wealth_quintile if time==2, contents(mean beta01_i p50 beta01_i mean beta02_i p50 beta02_i )  replace 
list   
twoway (line table1  wealth_quintile) (line table2  wealth_quintile) 	///
(line table3  wealth_quintile) (line table4  wealth_quintile)
graph export "q02b_coef.png", as(png) replace
restore

///Q2C: BETA QUINTILE///
sort newid02 time
bysort newid02: egen mean_ln_c = mean(lnc)
gen ab_beta01_i = abs(beta01_i)
gen ab_beta02_i = abs(beta02_i)
xtile beta01i_quintile = ab_beta01_i if model01 == 1, nquantiles(5)
xtile beta02i_quintile = ab_beta02_i if model01 == 1, nquantiles(5)
gen lnw = log(hh_wealth_2010)
	//Using Beta from Model01
forvalues q=1/5	{
	quietly estpost sum lnc lny lnw  if beta01i_quintile==`q' & time==2, detail
	est store q02c01_`q'
}
esttab q02c01_*, replace label  ///
cells("mean(pattern(1 1 1 1 1) fmt(3)) p50(pattern(1 1 1 1 1) fmt(3))") width(\hsize) ///
mtitles("Q1" "Q2" "Q3" "Q4" "Q5") title("Coefficients by Beta01 Quintile"\label{q02c01})

esttab q02c01_* using "q02c01.tex", replace label  ///
cells("mean(pattern(1 1 1 1 1) fmt(3)) p50(pattern(1 1 1 1 1) fmt(3))") width(\hsize) ///
mtitles("Q1" "Q2" "Q3" "Q4" "Q5") title("Coefficients by Beta01 Quintile"\label{q02c01})

preserve
table beta01i_quintile if time==2, contents(mean mean_ln_c p50 mean_ln_c)  replace 
list   
twoway (line table1  beta01i_quintile) (line table2  beta01i_quintile),	///
legend(order(1 "Mean Consumption" 2 "Median Consumption"))
graph export "q02c01_lnc.png", as(png) replace
restore

preserve
table beta01i_quintile if time==2, contents(mean mean_ln_y p50 mean_ln_y )  replace 
list   
twoway (line table1  beta01i_quintile) (line table2  beta01i_quintile),	///
legend(order(1 "Mean Income" 2 "Median Income"))
graph export "q02c01_lny.png", as(png) replace
restore

preserve
table beta01i_quintile if time==2, contents(mean lnw p50 lnw)  replace 
list   
twoway (line table1  beta01i_quintile) (line table2  beta01i_quintile),	///
legend(order(1 "Mean Wealth" 2 "Median Wealth"))
graph export "q02c01_lnw.png", as(png) replace
restore

	//Using Beta from Model02
forvalues q=1/5	{
	quietly estpost sum lnc lny lnw  if beta02i_quintile==`q' & time==2, detail
	est store q02c02_`q'
}
esttab q02c02_*, replace label  ///
cells("mean(pattern(1 1 1 1 1) fmt(4)) p50(pattern(1 1 1 1 1) fmt(4))") width(\hsize) ///
mtitles("Q1" "Q2" "Q3" "Q4" "Q5") title("Coefficients by Beta02 Quintile"\label{q02c02})

esttab q02c02_* using "q02c02.tex", replace label  ///
cells("mean(pattern(1 1 1 1 1) fmt(4)) p50(pattern(1 1 1 1 1) fmt(4))") width(\hsize) ///
mtitles("Q1" "Q2" "Q3" "Q4" "Q5") title("Coefficients by Beta02 Quintile"\label{q02c02})

preserve
table beta02i_quintile if time==2, contents(mean mean_ln_c p50 mean_ln_c)  replace 
list   
twoway (line table1  beta02i_quintile) (line table2  beta02i_quintile),	///
legend(order(1 "Mean Consumption" 2 "Median Consumption"))
graph export "q02c02_lnc.png", as(png) replace
restore

preserve
table beta02i_quintile if time==2, contents(mean mean_ln_y p50 mean_ln_y )  replace 
list   
twoway (line table1  beta02i_quintile) (line table2  beta02i_quintile),	///
legend(order(1 "Mean Income" 2 "Median Income"))
graph export "q02c02_lny.png", as(png) replace
restore

preserve
table beta02i_quintile if time==2, contents(mean lnw p50 lnw)  replace 
list   
twoway (line table1  beta02i_quintile) (line table2  beta02i_quintile),	///
legend(order(1 "Mean Wealth" 2 "Median Wealth"))
graph export "q02c02_lnw.png", as(png) replace
restore

*QUESTION 3

* Transform string to numbers
encode wave, gen(n_wave)

* Define panel data structure
xtset hh n_wave

*It is unbalanced. Balanced it
by hh: gen count = _N
drop if count == 1
drop count

*Get one value per HH and per year
*collapse (sum) ctotal inctotal, by(hh)

*Aggregate consumption by wave
egen agg_c  = sum(ctotal), by(n_wave)

*Convert to logs
gen log_c=log(ctotal)
gen log_y=log(inctotal)
gen log_C=log(agg_c)

gen d_c=log_c[_n]-log_c[n-1]
gen d_y=log_y[_n]-log_y[n-1]
gen d_C=log_C[_n]-log_C[n-1]

save "/Users/Pau_Belda/Documents/Uni/Màster IDEA/2nd year/Development/PS3/dataUGAA_T.dta"

*Regress
xtreg d_c d_y d_C,fe
est store B


* For urban people
keep if urban==1
xtreg d_c d_y d_C,fe
est store B
esttab B C using table1.tex

*For rural people
use "/Users/Pau_Belda/Documents/Uni/Màster IDEA/2nd year/Development/PS3/dataUGAA_T.dta", clear
keep if urban==0
xtreg d_c d_y d_C,fe
est store D
esttab D using table1.tex, append


