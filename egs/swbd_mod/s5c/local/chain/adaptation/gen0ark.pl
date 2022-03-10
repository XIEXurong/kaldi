($is_ark,$out_value,$sub_sampling) = @ARGV;

use POSIX;

if ($out_value eq "") {
	$out_value = 0;
}

while ($line = <STDIN>)
{
	chomp $line;
	@chunk = split(/\s+/, $line);
	$name = shift @chunk;
	print "$name";
    if ($is_ark eq "true")
    {
        print " [";
    }
    $frm_num=$chunk[0];
    if ($sub_sampling ne "" and $sub_sampling > 1)
    {
        $frm_num = POSIX::ceil($frm_num/$sub_sampling);
    }
	for ($i=0; $i<$frm_num; $i++)
	{
		print " $out_value";
	}
    if ($is_ark eq "true")
    {
        print " ]";
    }
	print "\n";
}


