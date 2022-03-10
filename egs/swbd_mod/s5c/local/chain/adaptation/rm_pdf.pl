#!/usr/bin/perl

# remove the label from a label file according to the scp file 
# e.g. perl ../scripts/rm_pdf.pl train_n1_clean_ali_stat.pdf feats_cv_train_n1_noise.scp > new_train_n1_clean_ali_stat.pdf

($pdfFile, $scpFile) = @ARGV;

%scpHash;

open(IN, $scpFile);

while($line = <IN>)
{
	@chunk = split(/\s+/, $line);
	$scpHash{$chunk[0]}=1;
}

close IN;

open(IN, $pdfFile);

while($line = <IN>)
{
	@chunk = split(/\s+/, $line);
	$scp = $scpHash{$chunk[0]};
	if($scp != 1)
	{
		print $line;
	}
}

close IN;