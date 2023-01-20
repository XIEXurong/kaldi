($out_value) = @ARGV;

if ($out_value eq "") {
	$out_value = 0;
}

while ($line = <STDIN>)
{
	chomp $line;
	@chunk = split(/\s+/, $line);
	$name = shift @chunk;
	print "$name";
	for ($i=0; $i<$chunk[0]; $i++)
	{
		print " $out_value";
	}
	print "\n";
}


