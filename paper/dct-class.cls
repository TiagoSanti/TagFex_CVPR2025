%-------------------------------------------------------------
% Essa classe foi criada por Sergio Roberto de Freitas
% e alterada por Erik Joey Hanashiro
% Em 01 de julho de 2001.
% Objetivo: padronizar as publicações produzidas no DCT-UFMS.
%-------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}[1995/12/01]

\LoadClass[a4paper,12pt]{report}[2000/05/19 v1.4b Standard LaTeX document class]
\RequirePackage[brazil]{babel}
%\RequirePackage[latin1]{inputenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{amsmath,amssymb,amsfonts,float,fancyhdr}





%-------------------------------------------------------------------------+
% macro para dvi e pdf
%\newif\ifpdf
%\ifx\pdfoutput\undefined
%   \pdffalse
%\else
%   \pdfoutput=1
%   \pdftrue
%\fi \ifpdf
%  \usepackage[pdftex]{graphicx}
%  \usepackage[pdftex,plainpages=true,colorlinks=true,
%              urlcolor=blue,
%              pdfstartview=FitH,
%              citecolor=blue,
%  ]{hyperref}
%\else
%\usepackage{graphicx}
%\usepackage[dvipdfm,colorlinks=true,
%              urlcolor=blue,
%              pdfstartview=FitH,
%              citecolor=blue,
%  ]{hyperref}
%\fi
%---------------- fim da macro para dvi e pdf -----------------------------
%--------- Configurando o layout da página ------------------------------------------------
\setlength{\topmargin}{-7mm}
\setlength{\textheight}{235mm}
\setlength{\oddsidemargin}{0mm}
\setlength{\evensidemargin}{0mm}
\setlength{\textwidth}{160mm}
% Se não quiser a identação no início do parágrafo, descomente a linha abaixo:
%\setlength{\parindent}{0em}
\setlength{\parskip}{1.5ex}
%\setlength{\headheight}{0mm}
%\addtolength{\headheight}{4pt} \tolerance=1000
\renewcommand{\chaptermark}[1]{%
\markboth{}{}}%
\lhead{\nouppercase{\quad\rightmark}\hfill{\sc dct-ufms}}%
\rhead{\nouppercase{}}%
\renewcommand{\headrulewidth}{0.7pt}%
\pagestyle{fancy}
%--------------------------------------------------------------------------
% Definindo e Customizando a Página Inicial 
\newcommand{\titulo}[1]{%
\begin{center}
\begin{minipage}[t]{12cm}
  \begin{center}
    {{\Large #1}}
  \end{center}
\end{minipage}
\end{center}
}
\newcommand{\autor}[1]{%
\relax\vskip 0.2cm\noindent
\begin{center}
\begin{minipage}[t]{10cm}
  \begin{center}
    {{\large #1}}
  \end{center}
\end{minipage}
\end{center}
}
\newcommand{\doctipo}[1]{%
\relax\vskip 0.2cm\noindent
\begin{center}
\begin{minipage}[t]{10cm}
  \begin{center}
    {{#1}} %\quad {(#2)}
  \end{center}
\end{minipage}
\end{center}
}

\newcommand{\orientacao}[1]{%
\vskip 2cm\noindent
\begin{center}{Orienta\c{c}\~ao:#1}\end{center}
}
%\newcommand{\doctipo}[1]{%
%\vskip 0.2cm\noindent
%\begin{center}{#1}\end{center}
%}
\newcommand{\docarea}[1]{%
\vskip 0.2cm\noindent
\begin{center}{\'{A}rea de Concentra\c{c}\~ao: #1}\end{center}
}
\newcommand{\textofree}[1]{%
%\vskip 1cm
\noindent
\begin{center}
{#1}
\end{center}
}
%--------------------------------------------------------------------------
\setcounter{tocdepth}{2}% estabelece a "profundidade dos bookmarks".

\newtheorem{theorem}{Teorema}[chapter]
\newtheorem{definition}{Definição}[chapter]
