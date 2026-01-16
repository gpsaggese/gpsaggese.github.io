pandoc gp_saggese_cv.md -o gp_saggese_cv.pdf --template=resume_template.tex --pdf-engine=xelatex

dockerized_latex.py -i gp_resume/gp_saggese_cv.tex -o gp_resume/gp_saggese_cv.pdf; open gp_resume/gp_saggese_cv.pdf
