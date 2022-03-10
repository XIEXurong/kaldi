#!/usr/bin/perl

# find the label from a label file according to the scp file 
# e.g. perl ../scripts/find_pdf.pl train_n1_clean_ali_stat.pdf feats_cv_train_n1_noise.scp > cv_n1_clean_ali_stat.pdf

($pdfFile, $scpFile) = @ARGV;

%pdfHash;

open(IN, $pdfFile);

while($line = <IN>)
{
	@chunk = split(/\s+/, $line);
	$pdfHash{$chunk[0]}=$line;
}

close IN;

open(IN, $scpFile);

while($line = <IN>)
{
	@chunk = split(/\s+/, $line);
	$pdf = $pdfHash{$chunk[0]};
	print $pdf;
}

close IN;