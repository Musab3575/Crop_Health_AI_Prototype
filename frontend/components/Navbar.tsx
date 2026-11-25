"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Leaf, Menu, X } from "lucide-react";
import { useState } from "react";

const links=[["Home","/"],["Dashboard","/dashboard"],["Scan leaf","/scanner"],["Classes","/supported"],["History","/history"],["Model","/about"]];

export default function Navbar(){
 const pathname=usePathname(); const [open,setOpen]=useState(false);
 return <header className="sticky top-0 z-50 border-b border-[#dfe8df] bg-[#f8faf6]/95 backdrop-blur"><nav className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 lg:px-8"><Link href="/" className="flex items-center gap-2 font-bold text-[#173f35]"><span className="grid h-9 w-9 place-items-center rounded-full bg-[#1d5a45] text-white"><Leaf size={19}/></span><span>Crop Health <span className="text-[#78a342]">AI</span></span></Link><div className="hidden items-center gap-1 md:flex">{links.map(([label,href])=>{const active=pathname===href; return <Link key={href} href={href} className={`rounded-full px-4 py-2 text-sm font-semibold ${active?"bg-[#e9f1e5] text-[#1d5a45]":"text-[#51655d] hover:text-[#173f35]"}`}>{label}</Link>})}</div><Link href="/scanner" className="hidden rounded-full bg-[#1d5a45] px-5 py-2.5 text-sm font-bold text-white shadow-sm md:block">Analyze a leaf</Link><button aria-label="Toggle menu" className="md:hidden" onClick={()=>setOpen(!open)}>{open?<X/>:<Menu/>}</button>{open&&<div className="absolute left-0 top-full flex w-full flex-col border-b bg-white p-4 md:hidden">{links.map(([label,href])=><Link key={href} href={href} onClick={()=>setOpen(false)} className="rounded-lg px-4 py-3 font-semibold text-[#51655d]">{label}</Link>)}</div>}</nav></header>
}
