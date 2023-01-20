($scp_file, $digit) = @ARGV;

%name_Hash;

open (IN, $scp_file);

while ($line = <IN>)
{
	$pos1 = index($line, " ");
	$pos2 = index($line, "\n");
	$name = substr($line, 0, $pos1);
	if ($name_Hash{$name} > 0)
	{
		$name_Hash{$name}++;
		$i = $name_Hash{$name};
		$content = substr($line, $pos1+1, $pos2 - $pos1);
        if ($digit ne "") {$i=(sprintf "%0".$digit."d", $i);}
        print "$name\#$i $content";
	}
	else
	{
		$name_Hash{$name} = 1;
		print $line;
	}
}

close IN;
