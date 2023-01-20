#!/usr/bin/perl
# usage: perl ../scripts/pdf2ark.pl dev01_ali_stat.pdf 3202 > 11111.ark

($pdfFile, $shift_offset) = @ARGV;

if ($shift_offset eq "")
{
	$shift_offset = 0;
}

open (IN, $pdfFile);

while ($line = <IN>)
{
	chomp($line);
	@chunk = split(/\s+/, $line);
	$file_name = shift @chunk;
	$row_num = @chunk;
	print "$file_name  \[ \n";
	for ($i = 0; $i < $row_num-1; $i++)
	{
		$tmp = $chunk[$i] + $shift_offset;
		print "  $tmp \n";
	}
	$tmp = $chunk[$row_num-1] + $shift_offset;
	print "  $tmp \]\n";
}

close IN;
